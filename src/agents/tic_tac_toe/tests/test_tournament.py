import pytest
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent

from agents.tic_tac_toe.tic_tac_toe.agent import root_agent


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


@pytest.mark.asyncio
async def test_happy_path():
    """Runs the agent on a simple input and expects a normal response."""
    user_input = "play"

    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="test_user"
    )
    content = UserContent(parts=[Part(text=user_input)])
    response = ""

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
        print(event)
        # if event.content.parts and event.content.parts[0].text:
        #     response = event.content.parts[0].text
        #     print(f"Response: {response}")

    # The answer in the input is wrong, so we expect the agent to provided a
    # revised answer, and the correct answer should mention engineering.
    # assert "data engineer" in response.lower()
    print(response.lower())
