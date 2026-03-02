from __future__ import annotations

import numpy as np
from functools import lru_cache


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

    def winner_if_play(pos: int, mark: int) -> int:
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
        if winner_if_play(int(a), -1) == -1:
            return int(a)

    # 2) block +1
    for a in legal:
        if winner_if_play(int(a), 1) == 1:
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


# ---------- FAST MINIMAX (memoized + alpha-beta) ----------

# Precomputed win lines in flat indices
_WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
)

# Encode each cell into base-3 digit:
# 0 -> empty, 1 -> X (+1), 2 -> O (-1)
def _encode_board(board: np.ndarray) -> int:
    code = 0
    p = 1
    # board length = 9
    for v in board:
        if v == 0:
            d = 0
        elif v == 1:
            d = 1
        else:  # -1
            d = 2
        code += d * p
        p *= 3
    return code


def _winner_from_board(board: np.ndarray) -> int:
    # returns +1 if X wins, -1 if O wins, else 0
    for a, b, c in _WIN_LINES:
        s = int(board[a] + board[b] + board[c])
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


@lru_cache(maxsize=None)
def _minimax_value(code: int, player: int, alpha: int, beta: int) -> int:
    """
    Returns game value from this state assuming perfect play.
      +1 => X eventually wins
       0 => draw
      -1 => O eventually wins

    player: +1 for X to move, -1 for O to move
    alpha/beta are in [-1, 1] for pruning
    """
    # Decode (only 9 cells; cheap)
    board = np.zeros(9, dtype=np.int8)
    x = code
    for i in range(9):
        d = x % 3
        x //= 3
        if d == 0:
            board[i] = 0
        elif d == 1:
            board[i] = 1
        else:
            board[i] = -1

    w = _winner_from_board(board)
    if w != 0:
        return w
    if np.all(board != 0):
        return 0

    legal = np.flatnonzero(board == 0)

    if player == 1:
        # X maximizes
        best = -1
        for a in legal:
            b2 = board.copy()
            b2[int(a)] = 1
            v = _minimax_value(_encode_board(b2), -1, alpha, beta)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
            if best == 1:
                break
        return best
    else:
        # O minimizes
        best = 1
        for a in legal:
            b2 = board.copy()
            b2[int(a)] = -1
            v = _minimax_value(_encode_board(b2), 1, alpha, beta)
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta:
                break
            if best == -1:
                break
        return best


def minimax_opponent(board: np.ndarray) -> int:
    """
    Fast optimal opponent for -1 (O).
    Uses memoized alpha-beta minimax with compact board encoding.
    """
    legal = np.flatnonzero(board == 0)
    if len(legal) == 0:
        return 0

    # O wants the smallest value (-1 best, 0 draw, +1 worst)
    best_move = int(legal[0])
    best_val = 2  # smaller is better for O

    for a in legal:
        b2 = board.copy()
        b2[int(a)] = -1
        val = _minimax_value(_encode_board(b2), 1, -1, 1)  # X to move next
        if val < best_val:
            best_val = val
            best_move = int(a)
            if best_val == -1:
                break

    return best_move