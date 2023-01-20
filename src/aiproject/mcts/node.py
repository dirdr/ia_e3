from collections import defaultdict
from typing import Self, Optional
import numpy as np
import aiproject.common.impl as common


class Node:
    """
    A node is a Game state
    this class is a wrapper for mcts implementation
    in addition to managing the common part,
    it generalizes important information for the tree
    """

    def __init__(self, board: np.ndarray, parent: Optional[Self] = None) -> None:
        """
        `childrens` is a list of currently knowed children
        `untried_action` will keep tracks of all actions that can be played from self,
            reaching to a new node
        `parent` is a reference to the parent node, that self is the child to.
            every node as a parent (except r) thats why it is typed as Otional
        """
        self.board = board
        self.childrens = []
        self.parent = parent
        self.n = 0
        self._result = defaultdict(int)
        self.untried_action = []

    @property
    def x_value(self) -> int:
        "return the mean of all game that passed through this node"
        return 0

    def is_terminal(self) -> bool:
        """
        A node is terminal if there are no action that can be played,
        no other state can be reached from this node
        """
        return common.is_over(self.board)

    def expand(self) -> Self:
        """
        expand self by one node
        if self is not terminal, take one action a
            consume a to create a child node and append it to the tree
        """
        action: int = self.untried_action.pop()
        copied: np.ndarray = self.board.copy()
        common.play_one_turn(copied, action)
        children: Self = Node(copied, self)
        self.childrens.append(children)
        return children

    def find_best_child_node(self, c: Optional[float] = 0.2) -> Self:
        """
        find and return the best child node according to the UCT formula

        @param c -> the balance factor, the value is set by default as 0.2
        """
        weights: list[float] = []
        for children in self.childrens:
            weights.append(
                (children.x_value / children.n)
                + c * np.sqrt((np.log(self.n) / children.n))
            )
        return self.childrens[np.argmax(weights)]

    def simulate(self) -> int:
        return 0

    def backpropagate(self, result: int) -> None:
        self.n += 1
