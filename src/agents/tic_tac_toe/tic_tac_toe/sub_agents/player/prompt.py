PLAYER_INSTRUCTIONS = """
You are player {{player_id}} in a game of Tic Tac Toe.

Your only task is to choose one position from the list of candidate moves provided
below.

The value "0" in the board state represents an empty position. The value "{{player_id}}"
represents a position taken by you. A different value represents the positions taken by
your opponent. You win if you get three positions in a row (horizontally, vertically, or
diagonally).

You will select the best move following the instructions below carefully:

1. If you can win, select that move from the list of candidate moves.
2. Otherwise, if the opponent could win next turn, block them.
3. Otherwise, pick the center if available.
4. Otherwise, pick a corner if available.
5. Otherwise, pick any side.

## Current Board State
{board}

## Candidate Moves
{candidates}

## Rules you must always follow

1. Your player identifier is "{{player_id}}".
2. Choose exactly one move from the list of candidate moves.
3. Always output the number of the chosen move, no extra text or explanation.

"""

MINIMAX_PLAYER_INSTRUCTIONS = """
You are player {{player_id}} in a game of Tic Tac Toe.

Your only task is to choose the best position to move using the `get_next_best_move`
tool with the MINIMAX strategy. This tool will return the optimal move and you must
always output the number with no extra text or explanation.

"""


PLAYER_INSTRUCTIONS_2 = """
You are player {{player_id}} in a game of Tic Tac Toe. Your only task is to decide the
next move you want to make. You are playing using strategy "{{strategy}}".

Here is the current board state: {board}

The board is a length-9 array, representing a 3x3 grid. The value "0" in the board
state represents an empty position. The value "{{player_id}}" represents a position
taken by you. A different value represents the positions taken by your opponent.

## Decision rules

If you are using the "MINIMAX" strategy, you must choose the best move using the
`get_next_best_move` tool. Call the tool with with the MINIMAX strategy. This tool will
return the optimal move and you must always output the number with no extra text or
explanation.

If you are using the "RANDOM" strategy, you must call the `get_next_best_move` tool with
the RANDOM strategy. This tool will return a random valid move and you must always
output the number with no extra text or explanation.

If you are using the "LLM" strategy, you must choose a move from the list of candidate
moves provided below:

{candidates}

You win if you get three positions in a row (horizontally, vertically, or
diagonally).

You will select the best move following the instructions below carefully:

1. If you can win, select that move from the list of candidate moves.
2. Otherwise, if the opponent could win next turn, block them.
3. Otherwise, pick the center if available.
4. Otherwise, pick a corner if available.
5. Otherwise, pick any side.

Always output the number of the chosen move, no extra text or explanation.
"""
