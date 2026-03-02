from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable

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
    board: shape (9,), values in {-1, 0, 1}
    Returns:
        1 if 1 player wins
        -1 if -1 player wins
        0 if neither has won
    """
    b = board.reshape(3,3)
    lines = []

    lines.extend(list(b))
    lines.extend(list(b.T))
    lines.append(np.diag(b))
    lines.append(np.diag(np.fliplr(b)))

    for line in lines:
        s = int(np.sum(line))
        if s==3:
            return 1
        if s==-3:
            return -1
    return 0


def is_draw(board: np.ndarray) -> bool:
    return np.all(board != 0) and check_winner(board) == 0


class TicTacToeEnv(gym.Env):
    """
    Single-agent tic tac toe environment:
        - Agent is always player 1
        - Opponent is always player -1
    
    Action space: Discrete(9) -> position 0..8

    Observation: Box(low = -1, high = 1, shape = (10,), dtype = np.int8)
        - First 9: board flattened
        - Last 1: current player to move (1 for agent turn, -1 for opponent turn)

    Rewards:
        +1 agent wins
        -1 agent loses
         0 draw
        -1 invalid move
        small step penalty to encourage fastest win possible 
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
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (10,), dtype = np.int8)

        self.board = np.zeros(9, dtype = np.int8)
        self.current_player = np.int8(1)    #agent playing
        self._last_info: Dict[str, Any] = {}
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.board, np.array([self.current_player], dtype = np.int8)])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)
        self.board[:] = 0
        self.current_player = np.int8(1)
        self._last_info = {"winner": 0, "draw": False, "invalid_move": False}
        if self.render_mode == "human":
            print(self.render())
        return self._get_obs(), self._last_info.copy()

    def step(self, action: int):
        action = int(action)
        info: Dict[str, Any] = {"winner": 0, "draw": False, "invalid_move": False}

        # Invalid move
        if action < 0 or action > 8 or self.board[action] != 0:
            info["invalid_move"] = True
            self._last_info = info
            return self._get_obs(), self.invalid_move_penalty, True, False, info

        # Agent places
        self.board[action] = 1

        # Check terminal after agent moves
        w = check_winner(self.board)
        if w == 1:
            info["winner"] = 1
            self._last_info = info
            return self._get_obs(), 1.0, True, False, info
        if is_draw(self.board):
            info["draw"] = True
            self._last_info = info
            return self._get_obs(), 0.0, True, False, info

        # Small step penalty for non-terminal valid move
        reward = self.step_penalty

        # Opponent move
        if self.opponent_policy is None:
            # If no policy, default to random
            legal = np.flatnonzero(self.board == 0)
            opp_action = int(self.np_random.choice(legal))
        else:
            opp_action = int(self.opponent_policy(self.board.copy()))
        
        # Safety: ensure opponent move is legal, random legal if not
        if opp_action < 0 or opp_action > 8 or self.board[opp_action] != 0:
            legal = np.flatnonzero(self.board == 0)
            opp_action = int(self.np_random.choice(legal))
        
        self.board[opp_action] = -1

        #Check terminal after opponent move
        w = check_winner(self.board)
        if w == -1:
            info["winner"] = -1
            self._last_info = info
            return self._get_obs(), -1.0, True, False, info
        if is_draw(self.board):
            info["draw"] = True
            self._last_info = info
            return self._get_obs(), 0.0, True, False, info
        
        if self.render_mode == "render":
            print(self.render())
        
        return self._get_obs(), 0.0, False, False, info
    
    def render(self):
        # Terminal friendly
        symbols = {1: "X", -1: "O", 0: " "}
        b = [symbols[int(v)] for v in self.board]
        rows = [
            f"{b[0]} | {b[1]} | {b[2]}",
            f"{b[3]} | {b[4]} | {b[5]}",
            f"{b[6]} | {b[7]} | {b[8]}",
        ]
        sep = "---+---+---"
        s = "\n".join([rows[0], sep, rows[1], sep, rows[2]])
        if self.render_mode == "human":
            print(s)
            return None
        return s

    def close(self): # A gymnasium thing, not really needed here
        pass    