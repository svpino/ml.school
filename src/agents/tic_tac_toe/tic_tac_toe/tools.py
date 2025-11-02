from google.adk.tools.tool_context import ToolContext

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


def tally(tool_context: ToolContext) -> str:
    """Tally the results of all games played so far."""
    state = tool_context.state
    tournament = state.get("tournament", [])

    player1_wins = tournament.count(1)
    player2_wins = tournament.count(2)
    draws = tournament.count(0)

    return (
        "Current Tournament Standings:\n"
        f"Total Games Played: {len(tournament)}\n"
        f"Player 1 Wins: {player1_wins}\n"
        f"Player 2 Wins: {player2_wins}\n"
        f"Draws: {draws}\n"
    )
