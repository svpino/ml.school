from pathlib import Path

import markdown
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from langchain_community.vectorstores import FAISS

from common.embeddings import CustomEmbeddingModel

from .prompts import FORMATTER_INSTRUCTIONS, RETRIEVER_INSTRUCTIONS

EMBEDDING_MODEL = "gemini/text-embedding-004"


def retrieve_content(tool_context: ToolContext, question: str) -> list[dict[str, str]]:  # noqa: ARG001
    """Retrieve documentation and reference materials to answer the question."""
    # Let's start by initializing embedding model we want to use.
    custom_embedding_model = CustomEmbeddingModel(model=EMBEDDING_MODEL)

    # We need to define the path where the vector store is located. To ensure the
    # code works regardless of where it's run from, we will use a path relative to
    # the location of this file.
    index_path = (
        Path(__file__).resolve().parents[3] / "data" / "index" / EMBEDDING_MODEL
    )

    # Now, we can load the vector store from disk. This vector store was created
    # by running the Indexing pipeline.
    vector_store = FAISS.load_local(
        str(index_path),
        custom_embedding_model,
        allow_dangerous_deserialization=True,
    )

    # Finally, we can run a similarity search to find the most relevant documents
    # related to the supplied question.
    results = vector_store.similarity_search(
        question,
        k=4,
    )

    return [
        {
            "file": result.metadata["file"],
            "content": result.page_content,
        }
        for result in results
    ]


def markdown_to_html(tool_context: ToolContext, text: str) -> str:  # noqa: ARG001
    """Convert the supplied Markdown text to HTML."""
    try:
        return markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "codehilite", "toc", "sane_lists"],
        )
    except Exception:
        # If the conversion fails for any reason, we will just use the original
        # answer.
        return text


def base_agent(model: str = "gemini/gemini-2.5-flash"):
    """Create the Retrieval-Augmented Generation agent."""
    retriever_agent = LlmAgent(
        model=LiteLlm(model=model),
        name="retriever",
        description="Answers user questions about the program.",
        instruction=RETRIEVER_INSTRUCTIONS,
        tools=[retrieve_content],
        output_key="answer_markdown",
    )

    formatter_agent = LlmAgent(
        model=LiteLlm(model=model),
        name="formatter",
        description="Formats answers from the retriever.",
        instruction=FORMATTER_INSTRUCTIONS,
        tools=[markdown_to_html],
        output_key="answer_html",
    )

    return SequentialAgent(
        name="workflow",
        sub_agents=[retriever_agent, formatter_agent],
        description=(
            "Executes a sequence of retrieval and formatting steps to answer questions."
        ),
    )


root_agent = base_agent()
