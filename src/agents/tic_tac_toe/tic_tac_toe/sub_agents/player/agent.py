import logging
import re

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from agents.tic_tac_toe.tic_tac_toe.sub_agents.player.prompt import (
    PLAYER_INSTRUCTIONS_2,
)
from agents.tic_tac_toe.tic_tac_toe.sub_agents.player.tools import get_next_best_move

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def player_output_guardrail(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Guardrail to validate the output from the model."""
    agent_name = callback_context.agent_name

    logger.info("Validating %s move...", agent_name)

    # Let's only worry about a text response coming from the model. In every other
    # case, we want to return and let the agent to proceed.
    if (
        llm_response.content
        and llm_response.content.parts
        and llm_response.content.parts[0].text
    ):
        response = llm_response.content.parts[0].text
    else:
        return None

    # At this point we want to check whether the response that came from the model is
    # a valid move.
    candidates = callback_context.state["positions"]
    logger.info("Candidates: %s", candidates)
    logger.info("Model response: %s", response)

    modified_response = False

    try:
        play = int(response)
    except (ValueError, TypeError):
        # If the model didn't answer with a valid number, we can try to extract one
        # from the reponse. This will fix any responses where the model presumably
        # returns a valid move but in an unexpected format.
        match = re.search(r"[1-9]", response or "")
        play = int(match.group()) if match else candidates[0]
        modified_response = True
        logger.info("Extracted move from response: %s", play)

    # Let's make sure the play is within the valid range of available moves.
    # If it is not, let's just pick the first available move.
    if play not in candidates:
        logger.warning(
            "Invalid move from %s: %s. Picking the first available move.",
            agent_name,
            play,
        )
        play = candidates[0]
        modified_response = True

    # If we had to modify the response from the model, we need to create and return
    # a new response object.
    if modified_response:
        return LlmResponse(
            content=types.Content(role="model", parts=str(play)),
            grounding_metadata=llm_response.grounding_metadata,
        )

    # If we reach this point, the move is valid so we want the agent call to proceed.
    return None


def board_after_agent_callback(callback_context: CallbackContext) -> LlmResponse | None:
    """Output the game board after every agent move."""
    # rows = callback_context.state["board"].split("\n")

    # for row in rows:
    #     print(" ".join(row))

    rows = [
        "".join(str(callback_context.state["board"][r * 3 + c]) for c in range(3))
        for r in range(3)
    ]
    print("\n".join(rows))

    return None


player1_agent = LlmAgent(
    model=LiteLlm(model="gemini/gemini-2.5-flash"),
    name="player1",
    description="Player 1",
    # instruction=MINIMAX_PLAYER_INSTRUCTIONS.replace("{{player_id}}", "1"),
    instruction=PLAYER_INSTRUCTIONS_2.replace("{{player_id}}", "1").replace(
        "{{strategy}}", "MINIMAX"
    ),
    output_key="play",
    after_agent_callback=board_after_agent_callback,
    after_model_callback=player_output_guardrail,
    tools=[get_next_best_move],
)

player2_agent = LlmAgent(
    model=LiteLlm(model="openai/gpt-5-mini"),
    name="player2",
    description="Player 2",
    instruction=PLAYER_INSTRUCTIONS_2.replace("{{player_id}}", "2").replace(
        "{{strategy}}", "RANDOM"
    ),
    output_key="play",
    after_agent_callback=board_after_agent_callback,
    after_model_callback=player_output_guardrail,
    tools=[get_next_best_move],
)
