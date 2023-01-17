import random
import numpy as np
from numba import njit

"""
every game data are stored inside 1 numpy array
np.ndarray (size 144) of uint8
B[0 - 63] list of possible moves
B[64 - 127] Gameboard (x, y) => 64 + x + 8*y
B[-1] : number of possibles moves
B[-2] : reserved
B[-3] : current player

There are two player, vertical (Player 0) and horizontal (Player 1)
Player 0 win => score : 1 
Player 1 win => score : -1
"""


STARTING_BOARD: np.ndarray = np.zeros(144, dtype=np.uint8)


@njit
def to_board_index(x_coordinate: int, y_coordinate: int) -> int:
    """
    convert (x, y) coordinate into gameboard index
    return the board index of the encoded coordinate
    """
    return 64 + 8 * y_coordinate + x_coordinate


@njit
def encode_play_id(player_id: int, x_coordinate: int, y_coordinate: int) -> int:
    "encode a play_id"
    return player_id * 100 + x_coordinate * 10 + y_coordinate


@njit
def decode_play_id(play_id: int) -> tuple[int, int, int]:
    """
    decode an encoded play_id,
    return a tuple [player_id, x_coordinate, y_coordinate]
    """
    return int(play_id / 100), (int(play_id / 10) % 10), play_id % 10


@njit
def is_over(board: np.ndarray) -> bool:
    "return true if the game is over else false"
    return board[-1] == 0


@njit
def get_winner(board: np.ndarray) -> int:
    """
    return the game winner if there is any,
    return 0 else
    """
    if board[-2] == 10:
        return 1
    if board[-2] == 20:
        return -1
    return 0


@njit
def _update_possible_moves(board: np.ndarray, player_id: int) -> None:
    """
    update a player list of possibles moves.
    (player_id = 0 => Vertical player, player_id = 1 => horizontal player)
    this function also update board[-1] which is the number of possibles move
    """
    count: int = 0
    x: int = 0
    y: int = 0
    offset: int = 0
    if player_id == 0:
        x = 8
        y = 7
        offset = 8
    else:
        x = 8
        y = 7
        offset = 1
    for x in range(x):
        for y in range(y):
            move_id = to_board_index(x, y)
            if board[move_id] == 0 and board[move_id + offset] == 0:
                board[count] = encode_play_id(player_id, x, y)
                count += 1
    board[-1] = count


@njit
def play_one_turn(board: np.ndarray, play_id: int) -> None:
    """
    play one game turn
    move id is the encoded move:
        this include the player that played the move,
        the coordinate of the play
    """
    player_id, x_coordinate, y_coordinate = decode_play_id(play_id)
    move_id: int = to_board_index(x_coordinate, y_coordinate)
    board[move_id] = 1
    if player_id == 0:
        board[player_id + 8] = 1
    else:
        board[player_id + 1] = 1
    next_player: int = 1 - player_id
    _update_possible_moves(board, next_player)
    board[-3] = next_player

    if is_over(board):
        # player 0 win => 10, player 1 win => 20
        board[-2] = (player_id + 1) * 10


@njit
def play_full_game(board: np.ndarray, move_id: int) -> None:
    "play an entire game"
    while not is_over(board):
        play_id: int = board[move_id]  # at index move_id is the encoded play_id
        play_one_turn(board, play_id)


@njit
def play_full_game_random(board: np.ndarray) -> None:
    "play an entire game with random move"
    while not is_over(board):
        move_id: int = random.randint(0, board[-1] - 1)
        play_id: int = board[move_id]
        play_one_turn(board, play_id)
