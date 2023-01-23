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
        self._n = 0
        self._result = defaultdict(int) # store the number of win and lose 
        self._untried_action = None

    @property
    def x_value(self) -> int:
        "return the mean of all game that passed through this node"
        return 0

    @property
    def n(self):
        return self._n

    @property
    def untried_action(self):
        if self._untried_action is None:
            self._untried_action = common.get_legal_actions(self.board)
        return self._untried_action

    def is_terminal(self) -> bool:
        """
        A node is terminal if there are no action that can be played,
        no other state can be reached from this node
        """
        return common.is_over(self.board)

    def can_expand(self) -> bool:
        """
            return true if the current node can have child s_i by consuming a_i
        """
        return len(self.untried_action) == 0

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

    def find_best_child(self, c: Optional[float] = 0.2) -> Self:
        """
        find and return the best child node according to the UCT formula

        @param c -> the balance factor, the value is set by default as 0.2
        """
        weights: list[float] = [] 
        modifier: int = 0
        if self.board[-3] == 0:
            modifier = 1
        else:   
            modifier = -1
        for children in self.childrens:
            weights.append(
                modifier * (children.x_value / children.n)
                + c * np.sqrt((np.log(self.n) / children.n))
            )
        return self.childrens[np.argmax(weights)]

    def simulate(self) -> int:
        """
        simulate a game from the current node (self) 
        the gme will be played with a random policy
        """
        simulation_board = self.board.copy()
        common.play_full_game_random(simulation_board)
        return common.get_winner(simulation_board)

    def backpropagate(self, result: int) -> None:
        """
        backpropagate the result (comming from a simulation) to the current node and its parent
        updating the number of time the node has been visited and the result.
        """
        self._n += 1
        self._result[result] += 1
        if self.parent:
            self.parent.backpropagate(result)
