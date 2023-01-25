from abc import ABC, abstractmethod
from aiproject.mcts.node import Node
from aiproject.mcts.search import Search
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

    def play(self, board: np.ndarray) -> None:
        move_id: int = self.find_best_move()
        play_id: int = board[move_id]
        common.play_one_turn(board, play_id)

    def __str__(self) -> str:
        return __class__.__name__


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
        self, player_0: Player, player_1: Player, number_of_match: int = 10
    ) -> None:
        self.number_of_match = number_of_match
        self.player_0 = player_0
        self.player_1 = player_1
        self.current_player = 0
        self.results = {}

    def get_current_player_instance(self) -> Player:
        if self.current_player == 0:
            return self.player_0
        return self.player_1

    def get_player_instance_by_result(self, result: int) -> Player:
        if result == 1:
            return self.player_0
        return self.player_1

    def battle(self) -> None:
        board: np.ndarray = STARTING_BOARD
        while not common.is_over(board):
            self.get_current_player_instance().play(board)
        result: int = common.get_winner(board)
        self.results[str(self.get_player_instance_by_result(result))] += 1
