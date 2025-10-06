PLAYER_INSTRUCTIONS = """
You are player {{player_id}} in a game of Tic Tac Toe.

Your only task is to choose one position from the list of available moves provided
below.

The value "0" in the board state represents an empty position. The value "{{player_id}}"
represents a position taken by you. A different value represents the positions of your
opponent. You win if you get three positions in a row (horizontally, vertically, or
diagonally).

Strategy priorities:
1. If you can win this turn, select that move from the list of available moves.
2. Otherwise, if the opponent could win next turn, block them.
3. Otherwise, pick the center if available.
4. Otherwise, pick a corner if available.
5. Otherwise, pick any side.

Current board state:
{board}

Available moves:
{moves}

Rules you must always follow:

1. Choose exactly one move from the list of available moves.
2. Always output the number of the chosen move, no extra text or explanation.

"""
