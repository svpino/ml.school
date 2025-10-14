# These are all the possible winning combinations on a board.
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


def get_winner(board: list[int]) -> int | None:
    """Return the identifier of the player who won the game or whether it's a draw.

    Args:
        board: The current state of the board.

    Returns:
        1 or 2 if either player has won, 0 for draw (full board, no winner), or None if
        the game is ongoing.

    """
    for a, b, c in _WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]

    if all(v != 0 for v in board):
        return 0

    return None
