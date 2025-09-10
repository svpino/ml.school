import asyncio
import random
import time

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai.errors import ServerError
from google.genai.types import Part, UserContent
from langchain_community.vectorstores import FAISS

from common.embeddings import CustomEmbeddingModel

from .prompts import INSTRUCTIONS

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


def base_agent(model: str = "gemini/gemini-2.5-flash"):
    """Create the RAG agent."""
    return LlmAgent(
        model=LiteLlm(model=model),
        name="rag",
        description="Answers user questions about the program.",
        instruction=INSTRUCTIONS,
        tools=[retrieve],
    )


class Agent:
    """A wrapper around the agent ."""

    def __init__(self, model, logger) -> None:
        """Initialize the agent."""
        self.runner = InMemoryRunner(agent=base_agent(model=model))
        self.logger = logger

    def run(self, question: str):
        """Run the agent to answer the supplied question."""
        return asyncio.run(
            self._agent_run(
                question=question,
                agent_timeout=120,
            )
        )

    async def _agent_run(self, question, agent_timeout):
        t = time.monotonic()
        message = UserContent(parts=[Part(text=question)])

        session = await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id="user"
        )

        # We want to keep trying to get a response from the agent until we either get
        # a response or we hit the timeout.
        while time.monotonic() - t < agent_timeout:
            try:
                async for event in self.runner.run_async(
                    user_id=session.user_id,
                    session_id=session.id,
                    new_message=message,
                ):
                    if event.content and event.content.parts:
                        if event.get_function_calls():
                            self.logger.info(
                                'Agent is calling the "%s" function...',
                                event.get_function_calls()[0].name,
                            )
                        elif event.get_function_responses():
                            self.logger.info(
                                'Agent received response from the "%s" function',
                                event.get_function_responses()[0].name,
                            )
                        elif event.is_final_response():
                            self.logger.info("Agent received the final response.")
                            return {
                                "status": "success",
                                "answer": event.content.parts[0].text,
                            }
            except ServerError:
                # If we get a server error, we want to wait a bit and then try again.
                # This will get around transient errors from the server when the model
                # is overloaded.
                await asyncio.sleep(10 + random.random() * 10)

        return {
            "status": "failed",
        }


root_agent = base_agent()
