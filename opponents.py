from __future__ import annotations
import numpy as np


def random_opponent(board: np.ndarray) -> int:
    legal = np.flatnonzero(board == 0)
    return int(np.random.choice(legal))


def heuristic_opponent(board: np.ndarray) -> int:
    """
    Simple opponent for -1 (O):
      1) Win if possible
      2) Block agent win
      3) Take center
      4) Take a corner
      5) Take any legal
    """
    b = board.reshape(3, 3)

    def winner_if_play(pos, mark):
        tmp = board.copy()
        tmp[pos] = mark
        bb = tmp.reshape(3, 3)
        lines = []
        lines.extend(list(bb))
        lines.extend(list(bb.T))
        lines.append(np.diag(bb))
        lines.append(np.diag(np.fliplr(bb)))
        for line in lines:
            s = int(np.sum(line))
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    legal = np.flatnonzero(board == 0)

    # 1) win as -1
    for a in legal:
        if winner_if_play(a, -1) == -1:
            return int(a)

    # 2) block +1
    for a in legal:
        if winner_if_play(a, 1) == 1:
            return int(a)

    # 3) center
    if board[4] == 0:
        return 4

    # 4) corners
    corners = [0, 2, 6, 8]
    corner_legal = [c for c in corners if board[c] == 0]
    if corner_legal:
        return int(np.random.choice(corner_legal))

    # 5) any
    return int(np.random.choice(legal))