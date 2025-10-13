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


def get_next_best_move(board: list[int], player: int) -> int:
    """Return the best move given the current state of the board.

    Args:
        board: Current state of the board.
        player: The player that is about to move.

    Returns:
        The index of board where the player should move.

    """
    best_score: tuple[int, int] | None = None
    best_index: int | None = None

    for index in _available_moves(board):
        board[index] = player
        score = _minimax(board=board, to_move=2 if player == 1 else 1, me=player)
        board[index] = 0

        if (
            best_score is None
            or score[0] > best_score[0]
            or (score[0] == best_score[0] and score[1] < best_score[1])
        ):
            best_score = score
            best_index = index

    return best_index


def _winner(board: list[int]) -> int | None:
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


def _available_moves(board: list[int]) -> list[int]:
    """Return the list of available moves on the board."""
    return [i for i, v in enumerate(board) if v == 0]


def _score(board: list[int], me: int) -> int:
    """Return the terminal score from the perspective of the `me` player.

    Args:
        board: The current state of the board.
        me: The player for whom we are calculating the score.

    Returns:
        1 if the player has won, 0 for a draw, -1 if the opponent has won, or None if
        the game is not over.

    """
    winner = _winner(board)

    # If the game is not over, return None.
    if winner is None:
        return None

    if winner == me:
        return 1

    if winner == 0:
        return 0

    # If we get to this point, my opponent won the game
    # so we need to return a low score.
    return -1


def _better(outcome1: tuple[int, int], outcome2: tuple[int, int]) -> bool:
    """Return whether an outcome is better than another.

    We always want to pick a winning outcome over a draw or losing outcome, and a draw
    over a losing outcome.

    If two outcomes are the same, we want to pick the one that reaches a win the soonest,
    or the one that delays a loss or draw the longest.

    Args:
        outcome1: A tuple of (score, steps) for the first outcome.
        outcome2: A tuple of (score, steps) for the second outcome.

    Returns:
        True if outcome1 is better than outcome2, False otherwise.

    """
    outcome1_score, outcome1_steps = outcome1
    outcome2_score, outcome2_steps = outcome2

    # If both outcomes are different, we always pick the one with the highest score.
    if outcome1_score != outcome2_score:
        return outcome1_score > outcome2_score

    # If both outcomes lead to a win, we want to pick the one with the fewer steps.
    if outcome1_score == 1:
        return outcome1_steps < outcome2_steps

    # If both outcomes lead to a loss, we want to pick the one that delays that loss
    # the longest.
    if outcome1_score == -1:
        return outcome1_steps > outcome2_steps

    # If both outcomes lead to a draw, we want to pick the one that delays that draw
    # the longest.
    return outcome1_steps > outcome2_steps


def _worse(outcome1: tuple[int, int], outcome2: tuple[int, int]) -> bool:
    """Return whether an outcome is worse than another.

    This function is the inverse of the `_better` function.
    """
    return _better(outcome2, outcome1)


def _minimax(board: list[int], to_move: int, me: int) -> tuple[int, int]:
    """Minimax algorithm to evaluate the best possible outcome for a `me` player.

    Args:
        board: The current state of the board.
        to_move: The player who is about to move.
        me: The player for whom we are calculating the outcome.

    Returns:
        A tuple of (score, steps) from 'me' perspective with optimal play. The number
        of steps indicates how many moves it takes to reach the end of the game.

    """
    # Let's compute the score of the player at this point and return it
    # if the game is over.
    score = _score(board, me)
    if score is not None:
        return (score, 0)

    next_player = 2 if to_move == 1 else 1
    outcomes: list[tuple[int, int]] = []

    for index in _available_moves(board):
        board[index] = to_move
        score, steps = _minimax(board, next_player, me)
        board[index] = 0

        # We want to add an extra step to account for the move we just made.
        outcomes.append((score, steps + 1))

    if to_move == me:
        # If we are evaluating the player we are interested in, we want
        # to find the best possible outcome.
        best_outcome = outcomes[0]
        for o in outcomes[1:]:
            if _better(o, best_outcome):
                best_outcome = o

        return best_outcome

    # If we are evaluating the opponent, we want to find the worst possible
    # outcome for them.
    worst_outcome = outcomes[0]
    for o in outcomes[1:]:
        if _worse(o, worst_outcome):
            worst_outcome = o

    return worst_outcome
