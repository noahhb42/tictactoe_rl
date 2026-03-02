from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


def check_winner(board: np.ndarray) -> int:
    """
    board: shape (9,), values in {-1, 0, +1}
    Returns:
      +1 if +1 player wins
      -1 if -1 player wins
       0 otherwise
    """
    b = board.reshape(3, 3)
    lines = []
    lines.extend(list(b))            # rows
    lines.extend(list(b.T))          # cols
    lines.append(np.diag(b))         # diag
    lines.append(np.diag(np.fliplr(b)))

    for line in lines:
        s = int(np.sum(line))
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def is_draw(board: np.ndarray) -> bool:
    return np.all(board != 0) and check_winner(board) == 0


class TicTacToeEnv(gym.Env):
    """
    Single-agent TicTacToe environment:
      - Agent always plays as +1 (X)
      - Opponent plays as -1 (O) using an injected policy

    Action space: Discrete(9) -> position 0..8
    Observation: Box(low=-1, high=1, shape=(10,), dtype=int8)
      - first 9: board flattened
      - last 1: current player to move (+1 for agent turn, -1 for opponent turn)
        (agent always acts as +1 in this environment, but we keep it explicit)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        opponent_policy: Optional[Callable[[np.ndarray], int]] = None,
        step_penalty: float = -0.01,
        invalid_move_penalty: float = -1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.opponent_policy = opponent_policy
        self.step_penalty = float(step_penalty)
        self.invalid_move_penalty = float(invalid_move_penalty)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.int8
        )

        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = np.int8(1)  # +1 = agent, -1 = opponent
        self._last_info: Dict[str, Any] = {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.board, np.array([self.current_player], dtype=np.int8)])

    def action_masks(self) -> np.ndarray:
        """
        For sb3-contrib MaskablePPO.
        True for legal actions, False for illegal actions.
        """
        return (self.board == 0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.board[:] = 0
        self.current_player = np.int8(1)
        self._last_info = {"winner": 0, "draw": False, "invalid_move": False}
        return self._get_obs(), self._last_info.copy()

    def step(self, action: int):
        action = int(action)
        info: Dict[str, Any] = {"winner": 0, "draw": False, "invalid_move": False}

        # Invalid move (should be prevented by masking during training, but keep for safety)
        if action < 0 or action > 8 or self.board[action] != 0:
            info["invalid_move"] = True
            self._last_info = info
            return self._get_obs(), self.invalid_move_penalty, True, False, info

        # Agent places X (+1)
        self.board[action] = 1

        # Check terminal after agent move
        w = check_winner(self.board)
        if w == 1:
            info["winner"] = 1
            self._last_info = info
            return self._get_obs(), 1.0, True, False, info
        if is_draw(self.board):
            info["draw"] = True
            self._last_info = info
            return self._get_obs(), 0.0, True, False, info

        reward = self.step_penalty

        # Opponent move
        if self.opponent_policy is None:
            legal = np.flatnonzero(self.board == 0)
            opp_action = int(self.np_random.choice(legal))
        else:
            opp_action = int(self.opponent_policy(self.board.copy()))

        # Safety: enforce legal opponent move
        if opp_action < 0 or opp_action > 8 or self.board[opp_action] != 0:
            legal = np.flatnonzero(self.board == 0)
            opp_action = int(self.np_random.choice(legal))

        self.board[opp_action] = -1

        # Check terminal after opponent move
        w = check_winner(self.board)
        if w == -1:
            info["winner"] = -1
            self._last_info = info
            return self._get_obs(), -1.0, True, False, info
        if is_draw(self.board):
            info["draw"] = True
            self._last_info = info
            return self._get_obs(), 0.0, True, False, info

        self._last_info = info
        return self._get_obs(), reward, False, False, info

    def render(self):
        symbols = {1: "X", -1: "O", 0: " "}
        b = [symbols[int(v)] for v in self.board]
        rows = [
            f" {b[0]} | {b[1]} | {b[2]} ",
            f" {b[3]} | {b[4]} | {b[5]} ",
            f" {b[6]} | {b[7]} | {b[8]} ",
        ]
        sep = "---+---+---"
        s = "\n".join([rows[0], sep, rows[1], sep, rows[2]])

        if self.render_mode == "human":
            print(s)
            return None
        return s

    def close(self):
        pass