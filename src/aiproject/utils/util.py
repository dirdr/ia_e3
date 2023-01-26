import numpy as np
import aiproject.common.impl as common

DEBUG: bool = True

def print_board(board):
    current_player: int = board[-3]
    print(f"Current player = {current_player}")
    for yy in range(8):
        y = 7 - yy
        s = str(y)
        for x in range(8):
            if board[common.to_board_index(x, y)] == 1:
                s += "::"
            else:
                s += "[]"
        print(s)
    s = " "
    for x in range(8):
        s += str(x) + str(x)
    print(s)

def print_move_line(play_id: int) -> None:
    player, x, y = common.decode_play_id(play_id)
    print(f"Player {player} ({x},{y})\n")
