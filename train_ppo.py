from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable

from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from tictactoe_env import TicTacToeEnv
from opponents import random_opponent, heuristic_opponent, minimax_opponent


def mask_fn(env: TicTacToeEnv):
    return env.action_masks()


def make_opponent(name: str) -> Callable:
    if name == "random":
        return random_opponent
    if name == "heuristic":
        return heuristic_opponent
    if name == "minimax":
        return minimax_opponent
    if name == "mixed":
        # Mix all three so the policy generalizes:
        # random makes it learn winning fast, minimax makes it learn not to lose.
        def mixed(board):
            r = random.random()
            if r < 0.40:
                return random_opponent(board)
            elif r < 0.80:
                return heuristic_opponent(board)
            else:
                return minimax_opponent(board)
        return mixed
    raise ValueError("opponent must be one of: random, heuristic, minimax, mixed")


def make_env(opponent_name: str):
    opp = make_opponent(opponent_name)

    def _thunk():
        env = TicTacToeEnv(opponent_policy=opp, step_penalty=-0.01, invalid_move_penalty=-1.0)
        env = ActionMasker(env, mask_fn)
        return env

    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--opponent", choices=["random", "heuristic", "minimax", "mixed"], default="mixed")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-path", type=str, default="models/maskppo_tictactoe")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(make_env(args.opponent), n_envs=args.n_envs)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(str(save_path))

    vec_env.close()
    print(f"Saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()