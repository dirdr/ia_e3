from abc import ABC, abstractmethod
from collections import defaultdict
from aiproject.mcts.node import Node
from aiproject.mcts.search import Search
import numpy as np
import aiproject.mc.monte_carlo as mc
import aiproject.common.impl as common
import aiproject.utils.util as u


STARTING_BOARD = np.zeros(144, dtype=np.uint8)
common._update_possible_moves(STARTING_BOARD, 0)


class Player(ABC):
    """
    an implementation of the player abstract class MUST override the play method with the AI way of finding the move
    the play method should return the board in the state s1 from s0 when a1 is played by the player
    """

    def __init__(self, id: int) -> None:
        self.id = id

    def get_keystring(self) -> str:
        """the player keystring is used to store result"""
        return f"{self.__class__.__name__}-{self.id}"

    @abstractmethod
    def play(self, board: np.ndarray) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}-{self.id}"


class MonteCarloPlayer(Player):
    def __init__(self, id: int, number_of_game_per_move: int) -> None:
        super().__init__(id)
        self.number_of_game_per_move = number_of_game_per_move

    def play(self, board: np.ndarray) -> np.ndarray:
        move_id: int = mc.find_best_move(board, self.number_of_game_per_move)
        play_id: int = board[move_id]
        common.play_one_turn(board, play_id)
        return board


class MonteCarloTreeSearchPlayer(Player):
    def __init__(self, id: int, simulation_time_in_s: float = 0.1) -> None:
        super().__init__(id)
        self.simulation_time_in_s = simulation_time_in_s

    def play(self, board: np.ndarray) -> np.ndarray:
        return Search(root=Node(board)).play(self.simulation_time_in_s)


class Battle:
    """
    the battle class is an interface for making battle between Ai:
        Monte carlo and Monte carlo tree search

    when creating a Player,
    id is not important for the game, its used to store games result and print them
    """

    def __init__(
        self, player_0: Player, player_1: Player, number_of_match: int = 10
    ) -> None:
        self.number_of_match = number_of_match
        self.player_0 = player_0
        self.player_1 = player_1
        self.current_player = 0
        self.results = defaultdict(int)

    def get_current_player_instance(self) -> Player:
        if self.current_player == 0:
            return self.player_0
        return self.player_1

    def get_player_instance_by_result(self, result: int) -> Player:
        "by convention when player 0 win, the score is 1 and when player 1 win, score is -1"
        if result == 1:
            return self.player_0
        return self.player_1

    def get_player_id_by_result(self, result: int) -> int:
        "by convention when player 0 win, the score is 1 and when player 1 win, score is -1"
        if result == 1:
            return 0
        return 1

    def full_battle(self) -> None:
        for _ in range(self.number_of_match):
            self.battle()

    def battle(self) -> None:
        """
        play a full match between `player_0` and `player_1`
        """
        board: np.ndarray = STARTING_BOARD.copy()
        self.current_player = 0
        while not common.is_over(board):
            board = self.get_current_player_instance().play(board)
            self.current_player = 1 - self.current_player
        result: int = common.get_winner(board)
        self.results[self.get_player_instance_by_result(result).get_keystring()] += 1

    def get_result_pretty_string(self) -> str:
        return f"{(self.results[self.player_0.get_keystring()] / self.number_of_match) * 100}% Win {self.player_0} - {(self.results[self.player_1.get_keystring()] / self.number_of_match) * 100}% Win {self.player_1}"
