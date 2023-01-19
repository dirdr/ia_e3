from collections import defaultdict
from common import impl as node
import numpy as np
from typing import Optional, Self

class MCTS:

    def __init__(self, c: float = 0.2, root) -> None:
        self.c = c  # coefficient that balance exploration and exploitation

    def get_best_move(self, simulation_number: Optional[int] = None) ->  


class Node:

    def __init__(self, board: np.ndarray, parent: Optional[Self]=None) -> None:
        self.board = board
        self.childrens = []
        self.parent = parent
        self.n = 0
        self._result = defaultdict(int)
    
    @property
    def x_value(self) -> int:
        "return the mean of all game that passed through this node"
        return self

    def find_childrens(self) -> Optional[set[Self]]:
        """
        this function will return a list of children for self
        """
        return None

    def select(self):
        """
        this function will perform the selection process

        """
        pass

    def find_best_child_node(self, c: Optional[float] = 0.2) -> Self:
        """
        find and return the best child node according to the UCT formula

        @param c -> the balance factor, the value is set by default as 0.2
        """
        weights: list[float] = []
        for children in self.childrens:
            weights.append(
                (children.x_value / children.n) + c * np.sqrt((np.log(self.n) / children.n))
            )
        return self.childrens[np.argmax(weights)]

    def backpropagate(self, reward) -> None:
        self.n += 1
