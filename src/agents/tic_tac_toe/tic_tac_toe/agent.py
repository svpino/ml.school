import logging

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from agents.tic_tac_toe.tic_tac_toe.prompt import TOURNAMENT_INSTRUCTIONS
from agents.tic_tac_toe.tic_tac_toe.sub_agents.commentator.agent import (
    commentator_agent,
)
from agents.tic_tac_toe.tic_tac_toe.sub_agents.game.agent import Game
from agents.tic_tac_toe.tic_tac_toe.sub_agents.player.agent import (
    player1_agent,
    player2_agent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tally(tool_context: ToolContext) -> str:
    """Tally the results of all games played so far."""
    logger.info("tally function invoked...")

    state = tool_context.state
    tournament = state.get("tournament", [])

    player1_wins = tournament.count(1)
    player2_wins = tournament.count(2)
    draws = tournament.count(0)

    return (
        "Current Tournament Standings:\n"
        f"Total Games Played: {len(tournament)}\n"
        f"Player 1 Wins: {player1_wins}\n"
        f"Player 2 Wins: {player2_wins}\n"
        f"Draws: {draws}\n"
    )


def after_game_callback(callback_context: CallbackContext) -> types.Content | None:
    """Keep track of the latest game results."""
    logger.info("after_game_callback invoked...")

    state = callback_context.state
    last_game_result = state.get("last_game_result")
    tournament = state.get("tournament", [])

    if last_game_result is not None:
        logger.info("Last game result: %s", last_game_result)

        if last_game_result == "PLAYER_1_WON":
            tournament.append(1)
        elif last_game_result == "PLAYER_2_WON":
            tournament.append(2)
        else:
            tournament.append(0)

    callback_context.state["tournament"] = tournament

    # We want to reset the result of the last game so that we don't accidentally
    # count it multiple times.
    callback_context.state["last_game_result"] = None


def get_tournament_agent() -> LlmAgent:
    """Create the Tournament Agent."""
    game_agent = Game(
        name="game",
        player1=player1_agent,
        player2=player2_agent,
        commentator=commentator_agent,
    )

    return LlmAgent(
        model=LiteLlm(model="gemini/gemini-2.5-flash"),
        name="tournament",
        description="Tournament agent",
        instruction=TOURNAMENT_INSTRUCTIONS,
        tools=[tally],
        sub_agents=[game_agent],
        after_agent_callback=after_game_callback,
    )


root_agent = get_tournament_agent()

APP_NAME = "story_app"
USER_ID = "12345"
SESSION_ID = "123344"


async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=root_agent,  # Pass the custom orchestrator agent
        app_name=APP_NAME,
        session_service=session_service,
    )
    return session_service, runner


async def call_agent_async(user_input_topic: str):
    session_service, runner = await setup_session_and_runner()

    current_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    if not current_session:
        logger.error("Session not found!")
        return

    content = types.Content(role="user", parts=[types.Part(text="Start")])
    events = runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content
    )

    final_response = "No final response captured."
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            # logger.info(
            #     f"Potential final response from [{event.author}]: {event.content.parts[0].text}"
            # )
            final_response = event.content.parts[0].text

    print("\n--- Agent Interaction Result ---")
    print("Agent Final Response: ", final_response)

    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("Final Session State:")
    import json

    print(json.dumps(final_session.state, indent=2))
    print("-------------------------------\n")


# asyncio.run(call_agent_async(""))
