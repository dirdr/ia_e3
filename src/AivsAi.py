from abc import ABC, abstractmethod
from aiproject.mcts.node import Node
from aiproject.mcts.search import Search
from typing import Self
import numpy as np
import aiproject.mc.monte_carlo as mc
import aiproject.common.impl as common

STARTING_BOARD = np.zeros(144, dtype=np.uint8)
common._update_possible_moves(STARTING_BOARD, 0)

class Player(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def find_best_move(self) -> int:
        """
        will find the best move for the current player and return it

        return an encoded move_id
        """
        pass

    def play(self, board: np.ndarray, move_id: int) -> None:
        common.play_one_turn(board, move_id)


class MonteCarloPlayer(Player):
    def __init__(self, board: np.ndarray, number_of_game_per_move: int) -> None:
        super().__init__()
        self.board = board
        self.number_of_game_per_move = number_of_game_per_move

    def find_best_move(self, board: np.ndarray) -> int:
        return mc.find_best_move(board, self.number_of_game_per_move, True)


class MonteCarloTreeSearchPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def find_best_move(self, board: np.ndarray) -> int:
        search: Search = Search(Node(board))
        return search.find_best_move(10)


class Battle:
    """
    the battle class is an interface for making battle between Ai:
        Monte carlo and Monte carlo tree search
    """

    def __init__(
        self, player_1: Player, player_2: Player, number_of_match: int = 10
    ) -> None:
        self.number_of_match = number_of_match
        self.player_1 = player_1
        self.player_2 = player_2
        self.result = []

    def battle() -> None:
        board: np.ndarray = common.START

