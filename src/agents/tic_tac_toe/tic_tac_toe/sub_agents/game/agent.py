import logging
import re
import time
from collections.abc import AsyncGenerator
from typing import override

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from agents.tic_tac_toe.tic_tac_toe.sub_agents.commentator.agent import (
    commentator_agent,
)
from agents.tic_tac_toe.tic_tac_toe.sub_agents.game.prompt import PLAYER_INSTRUCTIONS
from agents.tic_tac_toe.tic_tac_toe.sub_agents.game.tools import has_player_won

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
    moves = callback_context.state["moves"]
    modified_response = False

    try:
        play = int(response)
    except (ValueError, TypeError):
        # If the model didn't answer with a valid number, we can try to extract one
        # from the reponse. This will fix any responses where the model presumably
        # returns a valid move but in an unexpected format.
        match = re.search(r"[1-9]", response or "")
        play = int(match.group()) if match else 1
        modified_response = True

    # Let's make sure the play is within the valid range of available moves.
    # If it is not, let's just pick the first available move.
    if play < 1 or play > len(moves):
        play = 1
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
    rows = callback_context.state["board"].split("\n")

    for row in rows:
        print(" ".join(row))

    return None


player1_agent = LlmAgent(
    model=LiteLlm(model="gemini/gemini-2.5-flash"),
    name="player1",
    description="Player 1",
    instruction=PLAYER_INSTRUCTIONS.replace("{{player_id}}", "1"),
    output_key="play",
    after_agent_callback=board_after_agent_callback,
    after_model_callback=player_output_guardrail,
)

player2_agent = LlmAgent(
    model=LiteLlm(model="openai/gpt-5-mini"),
    name="player2",
    description="Player 2",
    instruction=PLAYER_INSTRUCTIONS.replace("{{player_id}}", "2"),
    output_key="play",
    after_agent_callback=board_after_agent_callback,
    after_model_callback=player_output_guardrail,
)


class Game(BaseAgent):
    """Custom agent for managing a game of Tic Tac Toe.

    This agent orchestrates a sequence of agents to manage the game state,
    validate player moves, and determine the outcome of the game.
    """

    player1: LlmAgent
    player2: LlmAgent
    commentator: LlmAgent
    board: list[int]

    def __init__(
        self,
        name: str,
        player1: LlmAgent,
        player2: LlmAgent,
        commentator: LlmAgent,
    ) -> None:
        """Initialize the game with two players and an empty board."""
        board = [0] * 9
        super().__init__(
            name=name,
            player1=player1,
            player2=player2,
            commentator=commentator,
            board=board,
            sub_agents=[player1, player2, commentator],
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        logger.info("[%s] Starting game...", self.name)

        result: str | None = None

        player_turn = 1
        moves = self._get_available_moves()

        while len(moves) > 0:
            state["board"] = self._get_board()
            state["moves"] = "\n".join(
                f"{i}. ({row}, {col})" for i, (row, col) in enumerate(moves, start=1)
            )

            player = self.player1 if player_turn == 1 else self.player2
            logger.info("%s's turn...", player.name)

            async for event in player.run_async(ctx):
                yield event

            player_move_index = int(state["play"])
            row, col = moves[player_move_index - 1]
            logger.info("%s's move: (%s, %s)", player.name, row, col)
            state["player"] = player_turn
            state["move"] = f"({row}, {col})"

            self.board[(row - 1) * 3 + (col - 1)] = player_turn
            state["board"] = self._get_board()

            async for event in self.commentator.run_async(ctx):
                yield event

            if has_player_won(self.board, player_turn):
                result = f"PLAYER_{player_turn}_WON"
                logger.info("Player %d has won!", player_turn)
                break

            moves = self._get_available_moves()
            player_turn = 2 if player_turn == 1 else 1

        if result is None:
            result = "DRAW"
            logger.info("The game is a draw.")

        event = self._create_event(result)
        await ctx.session_service.append_event(ctx.session, event)

        # import random

        # state["board"] = self._get_board()
        # state["moves"] = "\n".join(
        #     f"{i}. ({row}, {col})" for i, (row, col) in enumerate(moves, start=1)
        # )

        # player = self.player1 if player_turn == 1 else self.player2
        # logger.info("%s's turn...", player.name)

        # async for event in player.run_async(ctx):
        #     yield event

        # winner = random.choice([0, 1, 2])  # 0 for draw, 1 for player1, 2 for player2
        # if winner == 0:
        #     last_game_result = "DRAW"
        #     logger.info("The game is a draw.")
        # else:
        #     last_game_result = f"PLAYER_{winner}_WON"
        #     logger.info("Player %d has won!", winner)

        ######

    async def _mock_play(self, ctx: InvocationContext) -> AsyncGenerator[str, None]:
        import random

        winner = random.choice([0, 1, 2])
        return "DRAW" if winner == 0 else f"PLAYER_{winner}_WON"

    def _create_event(self, last_game_result: str) -> Event:
        """Create an event to update the system state at the end of the game."""
        state_delta = {
            "last_game_result": last_game_result,
        }

        return Event(
            invocation_id="game_ended",
            author=self.name,
            actions=EventActions(state_delta=state_delta),
            timestamp=time.time(),
        )

    def _get_board(self) -> str:
        r"""Return the board as a 3x3 string."""
        rows = ["".join(str(self.board[r * 3 + c]) for c in range(3)) for r in range(3)]
        return "\n".join(rows)

    def _get_available_moves(self) -> list[tuple[int, int]]:
        """Return available moves as (row, column), 1-based."""
        return [(i // 3 + 1, i % 3 + 1) for i, v in enumerate(self.board) if v == 0]


game_agent = Game(
    name="game",
    player1=player1_agent,
    player2=player2_agent,
    commentator=commentator_agent,
)
