import logging

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

from agents.tic_tac_toe.tic_tac_toe.sub_agents.player.prompt import (
    PLAYER_INSTRUCTIONS,
)
from agents.tic_tac_toe.tic_tac_toe.sub_agents.player.tools import (
    get_next_best_move,
    get_random_move,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def player_output_guardrail(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Guardrail to validate the output from the model.

    This guardrail will validate every move from the AI model and ensure they are
    only placed on valid/available board positions. If an invalid position is returned,
    this guardrail will automatically correct it to the first available candidate move.

    """
    agent_name = callback_context.agent_name
    logger.info("Running %s output guardrail...", agent_name)

    # Let's only worry about a model response with usable content.
    if not (
        llm_response
        and llm_response.content
        and llm_response.content.parts
        and llm_response.content.parts[0]
    ):
        logger.info("No usable content parts in response. Skipping guardrail.")
        return None

    part = llm_response.content.parts[0]

    # We are expecting the model to return structured output via a function call.
    # If that's not the case, let's just log a warning and skip the guardrail.
    if not (hasattr(part, "function_call") and part.function_call):
        logger.warning(
            "Model response doesn't look like a function call. Skipping guardrail."
        )
        return None

    # At this point, we have access to the arguments the model returned. These arguments
    # represent the structured output we are expecting from the model.
    args = getattr(part.function_call, "args", {}) or {}

    position = args.get("position")
    candidates = callback_context.state.get("positions", []) or []

    # We want to ensure the position returned by the model is one of the candidate
    # moves. If it's not, we will force the position to be the first candidate move.
    if candidates and position not in candidates:
        logger.warning(
            (
                'Invalid position "%s" returned by the model. '
                "Forcing response to position %s."
            ),
            position,
            candidates[0],
        )
        args["position"] = candidates[0]
        part.function_call.args = args

        # Here, we need to return a modified response with the corrected position.
        return LlmResponse(
            content=llm_response.content,
            grounding_metadata=llm_response.grounding_metadata,
        )

    # If we reach this point, the move is valid so we want the agent call to proceed.
    return None


class Turn(BaseModel):
    """Schema representing the structured output of a player turn."""

    player: int
    position: int
    strategy: str


def create_player_agent(player_id: int, model_name: str) -> LlmAgent:
    """Create a Tic-Tac-Toe player agent."""
    return LlmAgent(
        model=LiteLlm(model=model_name),
        name=f"player{player_id}",
        description=f"Tic Tac Toe Player {player_id}",
        instruction=PLAYER_INSTRUCTIONS.replace("{{player_id}}", str(player_id)),
        output_schema=Turn,
        output_key="turn",
        after_model_callback=player_output_guardrail,
        tools=[get_next_best_move, get_random_move],
    )


player1_agent = create_player_agent(1, "gemini/gemini-2.5-flash")
player2_agent = create_player_agent(2, "openai/gpt-5-mini")
