import numpy as np
from collections import defaultdict

class MCTS:

    def __init__(self, c: float = 0.2) -> None:
        """
        in an instance of this class,
        self will refer to the root node of the stats tree.
        """
        self.total_win = defaultdict(int)
        self.total_visit = defaultdict(int)
        self.children = dict()
        self.c = c # coefficient that balance exploration and exploitation

    def chose_best_node(self, node) -> dict:
        pass 

    def uct_child_selection(self, node):
        pass

    def simulate(self, node): 
        "this function will simulate a full game starting from node"
        
        

class Node:
    """
    In domineering, a node is a game board state
    """
