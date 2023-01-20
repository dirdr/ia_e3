from typing import Optional
from aiproject.mcts.node import Node


class Search:
    def __init__(self, root: Node) -> None:
        """
        create the search tree
        @param root -> the stats tree root node 'r'
        """
        self.root = root

    def get_best_move(self, simulation_number: int) -> int:
        """
        given that from root r, we can go to child c by doing action a,
        this function will return the best action to go to the calculated best child
        @param simulation_number -> number of rollout to go for before choosing the best child
        """
        for _ in range(0, simulation_number):
            pass
        return 0

    def selection(self):
        """
        start at ̀`self.root` and move down the tree until a leaf Node is found
        by traversing the tree:
            if the selected node is new: 
                make a rollout (simulation)
            if the selected node has been visited:
                fully expand that node with avaible actions,
                take the first new child and rollout from here
        """

    def tree_policy(self) -> None:
        """
        the tree policy is 
        """

