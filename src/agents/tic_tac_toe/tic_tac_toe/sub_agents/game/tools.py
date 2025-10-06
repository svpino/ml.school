def has_player_won(board: list[int], player: int) -> bool:
    """Return True if the given player has a winning line on the raw board."""
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


def _get_score(board: list[int], player: int, opponent: int) -> int | None:
    """Return 1 if player won, -1 if opponent won, 0 if draw, else None."""
    if has_player_won(board, player):
        return 1

    if has_player_won(board, opponent):
        return -1

    if all(v != 0 for v in board):
        return 0

    return None


def _get_available_moves(board: list[int]) -> list[int]:
    """Return the indices that are empty (contain 0)."""
    return [i for i, v in enumerate(board) if v == 0]


def _minimax(board: list[int], player_to_move: int, player: int, opponent: int) -> int:
    """Minimax evaluation from the perspective of `player`.

    Returns: 1 if winning, -1 if losing, 0 if draw with optimal play.
    """
    score = _get_score(board, player, opponent)
    if score is not None:
        return score

    next_player = 2 if player_to_move == 1 else 1
    best = -2 if player_to_move == player else 2

    for index in _get_available_moves(board):
        board[index] = player_to_move
        child_score = _minimax(board, next_player, player, opponent)
        board[index] = 0

        if player_to_move == player:
            best = max(best, child_score)
            if best == 1:
                break
        else:
            best = min(best, child_score)
            if best == -1:
                break

    return best


def get_best_strategy(board: list[int], player: int) -> tuple[int, int]:
    """Return the best strategy for a player given the current state of the board.

    Args:
        board: Current state of the board.
        player: The player that is about to move.

    Returns:
        A tuple (row, col) representing the best strategy for the player.

    """
    opponent = 2 if player == 1 else 1
    best_move = _get_available_moves(board)[0]

    # Let's initialize the best potential score with a very low value, so we can
    # replace it with any strategy that produces a better score.
    best_score = -2
    for index in _get_available_moves(board):
        # Let's now compute the best potential score we can have after making the
        # current move.
        board[index] = player
        score = _minimax(board, opponent, player, opponent)
        board[index] = 0

        if score > best_score:
            best_score = score
            best_move = index

            # If we find a move where we win, we don't need to keep looking for
            # a different move.
            if best_score == 1:
                break

    # Convert 0-based best_move to 1-based (row, col)
    row = best_move // 3 + 1
    col = best_move % 3 + 1
    return row, col
