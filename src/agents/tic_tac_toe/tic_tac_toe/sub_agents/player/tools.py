import random
from typing import Literal

_WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]

# The best strategy is to pick the center, then corners, then edges.
_TIEBREAK_ORDER = [4, 0, 2, 6, 8, 1, 3, 5, 7]


def _winner(board: list[int]) -> int | None:
    """Return 1 or 2 if either player has won, 0 for draw (full board, no winner), or None if game is ongoing."""
    for a, b, c in _WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if all(v != 0 for v in board):
        return 0  # draw
    return None


def _available_moves(board: list[int]) -> list[int]:
    return [i for i, v in enumerate(board) if v == 0]


def _terminal_score(board: list[int], me: int, opp: int) -> int | None:
    """Return 1 if 'me' wins, -1 if 'opp' wins, 0 if draw, or None if not terminal."""
    w = _winner(board)

    if w is None:
        return None

    if w == me:
        return 1

    if w == opp:
        return -1

    return 0


def _minimax(board: list[int], to_move: int, me: int, opponent: int) -> int:
    """Minimax evaluation from the perspective of a given player.

    Args:
        board: Current state of the board.
        to_move: The player whose turn it is to move.
        me: The player for whom we are evaluating the score.
        opponent: The opposing player.

    Returns:
        1 if forced win for me, -1 if forced loss for me, 0 if game will end in a draw.

    """
    score = _terminal_score(board, me, opponent)
    if score is not None:
        return score

    next_player = 2 if to_move == 1 else 1

    # We want to initialize the best score to a very low value if we want the current
    # player to win, or a very high value if we want the current player to lose.
    best_score = -2 if to_move == me else 2

    for index in _available_moves(board):
        board[index] = to_move
        score = _minimax(board, next_player, me, opponent)
        board[index] = 0

        # If we are evaluating the player we are interested in, we want
        # to maximize the score.
        if to_move == me:
            if score > best_score:
                best_score = score
                # We can't do better than winning, so let's break early.
                if best_score == 1:
                    break
        # If we are evaluating the opposing player, we want
        # to minimize the score.
        elif score < best_score:
            best_score = score
            # We can't do better than having the opposing player lose,
            # so let's break early.
            if best_score == -1:
                break

    return best_score


def get_next_best_move(
    board: list[int], player: int, strategy: Literal["MINIMAX", "RANDOM"]
) -> int:
    """Return the best move given the current state of the board.

    Args:
        board: Current state of the board.
        player: The player that is about to move.
        strategy: The strategy to use for selecting the next move.

    Returns:
        The index of board where the player should move.

    """
    if strategy == "RANDOM":
        result = random.choice(_available_moves(board))
        print("PICKED MOVE:", result)
        return result

    opponent = 2 if player == 1 else 1
    moves = _available_moves(board)

    scored_moves = []
    for index in moves:
        board[index] = player
        val = _minimax(board, opponent, player, opponent)
        board[index] = 0

        scored_moves.append((val, index))

    # Choose the move with the highest score; break ties using _TIEBREAK_ORDER
    best_score = max(val for val, _ in scored_moves)
    best_indices = [i for val, i in scored_moves if val == best_score]

    # Apply human-friendly tiebreak preference
    best_indices.sort(key=_TIEBREAK_ORDER.index)
    return best_indices[0]
