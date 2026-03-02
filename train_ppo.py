from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tictactoe_env import TicTacToeEnv
from opponents import random_opponent, heuristic_opponent


def make_env(opponent: str):
    if opponent == "random":
        opp = random_opponent
    elif opponent == "heuristic":
        opp = heuristic_opponent
    else:
        raise ValueError("opponent must be of form 'random' or 'heuristic'")
    
    def _thunk():
        return TicTacToeEnv(opponent_policy=opp, step_penalty=-0.01, invalid_move_penalty=-1.0)
    
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--opponent", choices=["random","heuristic"], default="random")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="models/ppo_tictactoe")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(make_env(args.opponent), n_envs=args.n_envs)

    model = PPO(
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
    print(f"Saved model to: {save_path}")


if __name__ == "__main__":
    main()