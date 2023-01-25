from numba.np.ufunc import parallel
import numpy as np
from numba import njit, prange
from aiproject.common import impl as common


@njit
def simulate_random_game(board: np.ndarray, move_id: int) -> np.ndarray:
    copied: np.ndarray = board.copy()
    play_id: int = copied[move_id]
    common.play_one_turn(copied, play_id)
    common.play_full_game_random(copied)
    return copied


@njit(parallel=True)
def simulate_random_games(
    board: np.ndarray, number_of_game: int, move_id: int
) -> np.float32:
    scores: np.ndarray = np.empty(number_of_game, dtype=np.int8)
    for game in prange(0, number_of_game):
        copied: np.ndarray = simulate_random_game(board, move_id)
        scores[game] = common.get_winner(
            copied
        )  # update the score for the game that just ended
    #print(scores)
    return scores.mean()


@njit
def find_best_move(board: np.ndarray, number_of_game: int) -> int:
    """
    simulate 'number_of_game' game per move for the two ia to chose the best move
    return the best move to play for the current ia
    """
    possible_moves_count: int = board[-1]
    means: np.ndarray = np.empty(possible_moves_count, dtype=np.float64)
    current_player: int = board[-3]
    for move_id in range(0, possible_moves_count):  # check all the possible move
        means[move_id] = simulate_random_games(board, number_of_game, move_id)
    if current_player == 1:
        return int(np.argmin(means))
    return int(np.argmax(means))
