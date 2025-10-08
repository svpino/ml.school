import pytest
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.tic_tac_toe.tic_tac_toe.agent import root_agent


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()


@pytest.fixture(scope="module")
def runner():
    return InMemoryRunner(agent=root_agent)


@pytest.fixture
async def create_session(runner):
    async def _make(state=None):
        return await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="test_user",
            state=state,
        )

    return _make


@pytest.mark.asyncio
async def test_agent_returns_tournament_tally(runner, create_session):
    session = await create_session()
    content = types.UserContent(parts=[types.Part(text="tally")])

    response = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
        if not getattr(event, "is_final_response", lambda: False)():
            continue

        if event.content and event.content.parts:
            part = event.content.parts[0]
            if getattr(part, "text", None):
                response = part.text
                break

    response = response.lower()

    assert "player 1 wins" in response
    assert "player 2 wins" in response
    assert "draws" in response


@pytest.mark.asyncio
async def test_agent_plays_game(runner, create_session):
    session = await create_session(state={"mode": "MOCK"})
    content = types.UserContent(parts=[types.Part(text="play")])

    async for _ in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=session.user_id, session_id=session.id
    )

    assert len(session.state["tournament"]) == 1
    assert session.state["last_game_result"] is None
