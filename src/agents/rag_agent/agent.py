from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from langchain_community.vectorstores import FAISS

from common.embeddings import CustomEmbeddingModel

from .prompts import INSTRUCTIONS

MODEL_NAME = "gemini/gemini-2.5-flash"
EMBEDDING_MODEL = "gemini/text-embedding-004"


def retrieve(question: str) -> list[dict[str, str]]:
    """Retrieve documentation and reference materials to answer the question."""
    # Let's start by initializing embedding model we want to use.
    custom_embedding_model = CustomEmbeddingModel(model=EMBEDDING_MODEL)

    # Now, we can load the vector store from disk. This vector store was created
    # by running the Indexing pipeline.
    vector_store = FAISS.load_local(
        f"data/index/{EMBEDDING_MODEL}",
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


root_agent = Agent(
    model=LiteLlm(model=MODEL_NAME),
    name="rag_agent",
    description="Answers user questions about the program.",
    instruction=INSTRUCTIONS,
    tools=[retrieve],
)
