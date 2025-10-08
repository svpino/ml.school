import logging
import random
import time
from collections.abc import AsyncGenerator
from typing import override

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from agents.tic_tac_toe.tic_tac_toe.tools.tools import has_player_won

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        mode = state.get("mode", "LIVE")

        logger.info("[%s] Starting game. Mode: %s", self.name, mode)
        result: str | None = None

        if mode == "MOCK":
            result = self._mock_play()
        else:
            player_turn = 1
            # available_positions = self._get_available_positions()
            state["positions"] = [i for i, v in enumerate(self.board) if v == 0]

            while len(state["positions"]) > 0:
                # state["board"] = self._get_board()
                state["board"] = self.board
                state["candidates"] = "\n".join(
                    f"{position}. ({position // 3 + 1}, {position % 3 + 1})"
                    for position in state["positions"]
                )

                player = self.player1 if player_turn == 1 else self.player2
                logger.info("%s's turn...", player.name)

                async for event in player.run_async(ctx):
                    yield event

                print("--------------------------------")
                print(f"{player.name} -> {state['play']}")
                print(state["board"])
                print(state["candidates"])

                # position = available_positions[int(state["play"]) - 1]
                # position = state["positions"][int(state["play"]) - 1]
                position = int(state["play"])

                print(
                    f"{player.name}'s move: {int(state['play'])}. ({position // 3 + 1}, {position % 3 + 1})"
                )
                print("--------------------------------")

                state["player"] = player_turn
                state["move"] = f"({position // 3 + 1}, {position % 3 + 1})"

                self.board[position] = player_turn
                state["positions"] = [i for i, v in enumerate(self.board) if v == 0]
                state["board"] = self.board

                async for event in self.commentator.run_async(ctx):
                    yield event

                if has_player_won(self.board, player_turn):
                    result = f"PLAYER_{player_turn}_WON"
                    logger.info("Player %d has won!", player_turn)
                    break

                # state["positions"] = self._get_available_positions()

                player_turn = 2 if player_turn == 1 else 1

            if result is None:
                result = "DRAW"
                logger.info("The game is a draw.")

        event = self._create_event(result)
        await ctx.session_service.append_event(ctx.session, event)

    def _mock_play(self) -> str:
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

    # def _get_available_positions(self) -> list[tuple[int, int]]:
    #     """Return the available positions in the board using (row, col) format."""
    #     return [(i // 3 + 1, i % 3 + 1) for i, v in enumerate(self.board) if v == 0]
