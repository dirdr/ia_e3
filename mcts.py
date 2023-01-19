from collections import defaultdict
from common import impl as node
import numpy as np

class MCTS:
    def __init__(self, c: float = 0.2, root) -> None:
        self.c = c  # coefficient that balance exploration and exploitation


class Node:

    def __init__(self, board: np.ndarray, parent=None) -> None:
        self.board = board
        self.children = []
        self.parent = parent
        self._n = 0
        self._result
