import logging

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm

from agents.tic_tac_toe.tic_tac_toe.sub_agents.commentator.prompt import (
    COMMENTATOR_INSTRUCTIONS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def commentator_after_agent_callback(
    callback_context: CallbackContext,
) -> LlmResponse | None:
    """Display the commentary from the commentator."""
    logger.info(callback_context.state["commentary"])

    return None


commentator_agent = LlmAgent(
    model=LiteLlm(model="openai/gpt-5-mini"),
    name="commentator",
    description="Game commentator",
    instruction=COMMENTATOR_INSTRUCTIONS,
    output_key="commentary",
    after_agent_callback=commentator_after_agent_callback,
)

agent = commentator_agent
