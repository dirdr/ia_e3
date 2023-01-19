import numpy as np
from numba import njit, prange
from common import impl as common

STARTING_BOARD = np.zeros(144, dtype=np.uint8)
common._update_possible_moves(STARTING_BOARD, 0)


@njit
def pvp_one_match(nog_player_0: int, nog_player_1: int, p=False) -> int:
    """
    play a full match between two ia
    depending on the current player, change simulation parameters
    """
    board: np.ndarray = STARTING_BOARD.copy()
    while not common.is_over(board):
        if board[-3] == 0:
            nog: int = nog_player_0
        else:
            nog: int = nog_player_1
        best_move: int = find_best_move(board, nog, p)
        play_id: int = board[best_move]
        common.play_one_turn(board, play_id)
    return common.get_winner(board)


@njit
def pvp_multiple_match(
    number_of_game: int, nog_player_0: int, nog_player_1: int, p=False
) -> np.ndarray:
    """
    play 'number of game' game between two ia
    return a ndarray where arr[i] = Player_i number of win, i ∈ {0, 1}
    """
    win_count: np.ndarray = np.zeros(2)
    for _ in range(number_of_game):
        winner: int = pvp_one_match(nog_player_0, nog_player_1, p=p)
        if winner == 1:
            win_count[0] += 1
        else:
            win_count[1] += 1
    return win_count


@njit
def simulate_random_game(board: np.ndarray, move_id: int) -> np.ndarray:
    copied: np.ndarray = board.copy()
    play_id: int = copied[move_id]
    common.play_one_turn(copied, play_id)
    common.play_full_game_random(copied)
    return copied


@njit(parallel=True)
def find_best_move(board: np.ndarray, number_of_game: int, p=False) -> int:
    """
    simulate 'number_of_game' game per move for the two ia to chose the best move
    return the best move to play for the current ia
    """
    possible_moves_count: int = board[-1]
    means: np.ndarray = np.zeros(possible_moves_count, dtype=np.float64)
    current_player: int = board[-3]
    for move_id in range(0, possible_moves_count):  # check all the possible move
        scores: np.ndarray = np.zeros(number_of_game, dtype=np.int32)
        if p == True:
            for game in prange(0, number_of_game):
                copied: np.ndarray = simulate_random_game(board, number_of_game)
                scores[game] = common.get_winner(
                    copied
                )  # update the score for the game that just ended
        else:
            for game in range(0, number_of_game):
                copied: np.ndarray = simulate_random_game(board, number_of_game)
                scores[game] = common.get_winner(
                    copied
                )  # update the score for the game that just ended
        means[move_id] = scores.mean()
    if current_player == 1:
        return int(np.argmin(means))
    return int(np.argmax(means))


def get_score_pretty_string(score: np.ndarray, number_of_game: int) -> str:
    return f"{(score[0] / number_of_game) * 100}% Win IA 1 - {(score[1] / number_of_game) * 100}% Win IA 2"
