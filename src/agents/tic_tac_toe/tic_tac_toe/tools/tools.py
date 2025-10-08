def has_player_won(board: list[int], player: int) -> bool:
    """Return whether the given player has a winning line on the raw board.

    Args:
        board: Current state of the board.
        player: The player to check for a winning line.

    Returns:
        True if the player has a winning line, otherwise False.

    """
    winning_lines = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )
    return any(
        board[i] == player and board[j] == player and board[k] == player
        for i, j, k in winning_lines
    )
