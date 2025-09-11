import asyncio
import random
import time

from google.adk.runners import InMemoryRunner
from google.genai.errors import ServerError
from google.genai.types import Part, UserContent
from metaflow import Config, Parameter, card, step

from agents.rag.agent import base_agent
from common.pipeline import Pipeline


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
                                'Agent "%s" calling tool "%s"...',
                                event.author,
                                event.get_function_calls()[0].name,
                            )
                        if event.is_final_response() and event.author == "formatter":
                            # At this point, we have the final response from the agent,
                            # so we can break the loop.
                            self.logger.info("Received final response.")

                session = await self.runner.session_service.get_session(
                    app_name=self.runner.app_name, user_id="user", session_id=session.id
                )

                if "answer_html" in session.state or "answer_markdown" in session.state:
                    return {
                        "status": "success",
                        "answer": session.state.get(
                            "answer_html", session.state.get("answer_markdown", "")
                        ),
                    }
            except ServerError:
                # If we get a server error, we want to wait a bit and then try again.
                # This will get around transient errors from the server when the model
                # is overloaded.
                await asyncio.sleep(10 + random.random() * 10)

        return {
            "status": "failed",
        }


def read_template(html):
    """Parse the supplied HTML template.

    This function is used to read an HTML template from a Metaflow Config
    parameter and return it as a dictionary that can be used by the card
    decorator.
    """
    return {"html": html}


class Rag(Pipeline):
    """A Metaflow pipeline that answers questions using a RAG agent."""

    model = Parameter(
        name="model",
        help="The underlying model that will be used by the agent.",
        default="gemini/gemini-2.5-flash",
    )

    template = Config("template", default="config/rag.html", parser=read_template)

    @card
    @step
    def start(self):
        """Start the pipeline by defining the questions we want to ask."""
        self.questions = [
            "Summarize how Metaflow branches work.",
            "How do I run the Training pipeline locally?",
            "Where can I find the code for the Training pipeline?",
        ]

        # For each question, we will use the agent to get an answer.
        self.next(self.answer_question, foreach="questions")

    @step
    def answer_question(self):
        """Run the agent to answer the question assigned to this branch."""
        # Let's create an instance of the agent that we want to use to answer the
        # question and initialize it with the supplied model.
        agent = Agent(model=self.model, logger=self.logger)

        self.question = self.input
        self.response = agent.run(question=self.question)

        self.status = self.response["status"]
        self.answer = self.response.get("answer", "")

        self.next(
            {
                "success": self.success,
                "failed": self.failed,
            },
            condition="status",
        )

    @card(type="html")
    @step
    def success(self):
        """Showcase the question and the generated answer in a Metaflow card."""
        self.html = (
            self.template["html"]
            .replace("[[QUESTION]]", self.question)
            .replace("[[ANSWER]]", self.answer)
        )

        self.next(self.join)

    @step
    def failed(self):
        """Handle any failures while running the agent."""
        self.logger.info('Failed to answer question "%s".', self.question)
        self.next(self.join)

    @card
    @step
    def join(self, inputs):
        """Join parallel branches."""
        self.responses = [
            {
                "question": i.question,
                "answer": i.answer,
                "status": i.status,
            }
            for i in inputs
        ]

        self.next(self.end)

    @step
    def end(self):
        """End the pipeline by printing the final response."""
        self.logger.info("Number of responses: %s", len(self.responses))


if __name__ == "__main__":
    Rag()
