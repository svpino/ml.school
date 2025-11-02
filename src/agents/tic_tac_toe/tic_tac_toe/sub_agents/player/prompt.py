PLAYER_INSTRUCTIONS = """
You are player {{player_id}} in a game of Tic Tac Toe. Your only task is to decide the
next move you want to make. If you haven't been told otherwise, you are playing
using the "MODEL" strategy.

Here is the current board state: {board}

The board is a length-9 array, representing a 3x3 grid. The value "0" in the board
state represents an empty position. The value "{{player_id}}" represents a position
taken by you. A different value represents the positions taken by your opponent.

You will follow the decision rules below to pick your next move. You will always
output a JSON object in the following format:

{
    "player": "{{player_id}}",
    "position": <position of the move you want to make>,
    "strategy": <the strategy you are using, one of "MINIMAX", "RANDOM", or "MODEL">
}

Here is the list of candidate moves you can pick from:
{candidates}

You can never pick a position that is not in the list of candidate moves.

## Decision Rules

### If you are using the "MINIMAX" strategy

You must choose the best move using the `get_next_best_move` tool. This tool will
return the best move for you to make, based on the current board state.


### If you are using the "RANDOM" strategy

You must choose the best move using the `get_random_move` tool. This tool will
return a random move for you to make, based on the current board state.


### If you are using the "MODEL" strategy

You must choose a random move from the list of candidate moves provided above. Follow
the guidelines below to pick the best move:

1. You win if you get 3 positions in a row (horizontally, vertically, or diagonally).
2. If you can win, select that move from the list of candidate moves.
3. Otherwise, if the opponent could win next turn, block them.
4. Otherwise, pick the center if available.
5. Otherwise, pick a corner if available.
6. Otherwise, pick any side.

"""
