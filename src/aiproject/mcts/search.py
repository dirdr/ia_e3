import numpy as np
from typing import Optional
from aiproject.mcts.node import Node


class Search:
    def __init__(self, root: Node) -> None:
        """
        create the search tree
        @param root -> the stats tree root node 'r'
        """
        self.root = root

    def play(self, simulation_number: int) -> np.ndarray:
        """
        given that from root r, we can go to child c by doing action a,
        this function will return the best action to go to the calculated best child
        @param simulation_number -> number of rollout to go for before choosing the best child
        """
        for _ in range(0, simulation_number):
            to_simulate: Node = self._selection()
            simulation_result = to_simulate.simulate()
            to_simulate.backpropagate(simulation_result)
        return self.root.find_best_child().board.copy()


    def _selection(self) -> Node:
        """
        the mcts selection (tree policy) describe how the algorithm should traverse the tree.
            - traversing through the best child until a non fully expanded child is found  
            if a leaf (or a non fully expanded node) is found, expand it by one child
        the tree policy will return the node to simulate:
            either the child resulting from the expansion of a node, or if the root is now a terminal node
        """
        current_node: Node = self.root
        while not current_node.is_terminal():
            if current_node.can_expand():
                return current_node.expand()
            else:
                current_node = current_node.find_best_child()
        return current_node
