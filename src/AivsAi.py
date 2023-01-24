from aiproject.mcts.search import Search
from typing import Self


class Battle:
    """
    the battle class is an interface for making battle between Ai:
        Monte carlo and Monte carlo tree search
    """

    def __init__(self, number_of_match: int=10, ai_1: str, ai_2: str) -> None:
        self.number_of_match = number_of_match

    def initialize_ai(self) -> None:
        pass
