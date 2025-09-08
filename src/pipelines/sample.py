import asyncio

from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent
from metaflow import FlowSpec, step

from agents.rag_agent.agent import root_agent


class RagAgent:
    def __init__(self):
        self.runner = InMemoryRunner(agent=root_agent)

    def run(self, question: str):
        return asyncio.run(self._agent_loop(loop_timeout=120, question=question))

    async def _agent_loop(self, loop_timeout, question):
        session = await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id="test_user"
        )

        content = UserContent(parts=[Part(text=question)])

        async for event in self.runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            print(event)
            if event.content.parts and event.content.parts[0].text:
                response = event.content.parts[0].text
                print(response)


class Sample(FlowSpec):
    """A Metaflow pipeline used for indexing the documentation of the project."""

    @step
    def start(self):
        agent = RagAgent()
        agent.run("Summarize how branches work")
        self.next(self.end)

    @step
    def end(self):
        print("Done")


if __name__ == "__main__":
    Sample()
