import logging

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from agents.tic_tac_toe.tic_tac_toe.prompt import TOURNAMENT_INSTRUCTIONS
from agents.tic_tac_toe.tic_tac_toe.sub_agents.game.agent import game_agent
from agents.tic_tac_toe.tic_tac_toe.tools import tally

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


root_agent = LlmAgent(
    model=LiteLlm(model="gemini/gemini-2.5-flash"),
    name="tournament",
    description="Tournament agent",
    instruction=TOURNAMENT_INSTRUCTIONS,
    tools=[tally],
    sub_agents=[game_agent],
    after_agent_callback=after_game_callback,
)
