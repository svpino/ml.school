import asyncio
import random
import time

from google.adk.runners import InMemoryRunner
from google.genai.errors import ServerError
from google.genai.types import Part, UserContent
from metaflow import step

from agents.rag_agent.agent import root_agent as agent
from common.pipeline import Pipeline


class Agent:
    """A wrapper around the RAG agent to use it in a Metaflow pipeline."""

    def __init__(self, logger) -> None:
        """Initialize the agent."""
        self.runner = InMemoryRunner(agent=agent)
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
                            return event.content.parts[0].text
            except ServerError:
                # If we get a server error, we want to wait a bit and then try again.
                # This will get around transient errors from the server when the model
                # is overloaded.
                await asyncio.sleep(10 + random.random() * 10)

        return None


class Rag(Pipeline):
    """A Metaflow pipeline that answers questions using a RAG agent."""

    @step
    def start(self):
        """Start the pipeline by defining the questions we want to ask."""
        self.questions = [
            "Summarize how Metaflow branches work.",
            "How do I run the Training pipeline locally?",
            "Where can I find the code for the Training pipeline?",
        ]

        # For each question, we will run the agent to get an answer.
        self.next(self.answer, foreach="questions")

    @step
    def answer(self):
        """Run the agent to answer the supplied question."""
        agent = Agent(logger=self.logger)

        self.question = self.input
        self.response = agent.run(question=self.question)

        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the parallel branches and create the final response."""
        self.answers = [(i.question, i.response) for i in inputs]

        self.response = ""
        for question, answer in self.answers:
            self.response += f"Question: {question}\nAnswer: {answer}\n\n"

        self.next(self.end)

    @step
    def end(self):
        """End the pipeline by printing the final response."""
        self.logger.info("%s", self.response)


if __name__ == "__main__":
    Rag()
