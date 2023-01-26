import numpy as np
import aiproject.common.impl as common

DEBUG: bool = True

def print_board(board):
    current_player: int = board[-3]
    for yy in range(8):
        y = 7 - yy
        s = str(y)
        for x in range(8):
            if board[common.to_board_index(x, y)] == 1:
                if current_player == 0:
                    s += ";;"
                else:
                    s += "::"
            else:
                s += "[]"
        print(s)
    s = " "
    for x in range(8):
        s += str(x) + str(x)
    print(s)

    nbMoves = board[-1]
    print("Possible moves :", nbMoves)
    s = ""
    for i in range(nbMoves):
        s += str(board[i]) + " "
    print(s)

def print_move_line(play_id: int) -> None:
    player, x, y = common.decode_play_id(play_id)
    print(f"Player {player} ({x},{y})")
