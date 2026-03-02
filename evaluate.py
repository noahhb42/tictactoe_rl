from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sb3_contrib import MaskablePPO

from tictactoe_env import TicTacToeEnv
from opponents import random_opponent, heuristic_opponent, minimax_opponent


@dataclass
class EvalStats:
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    invalid: int = 0
    total_reward: float = 0.0
    total_steps: int = 0


def get_opponent(name: str) -> Callable[[np.ndarray], int]:
    if name == "random":
        return random_opponent
    if name == "heuristic":
        return heuristic_opponent
    if name == "minimax":
        return minimax_opponent
    raise ValueError("opponent must be 'random', 'heuristic', or 'minimax'")


def run_episode(env: TicTacToeEnv, model: MaskablePPO, deterministic: bool) -> tuple[float, int, dict]:
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0
    last_info = info

    while not done:
        # Provide action masks (legal actions only)
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        ep_reward += float(reward)
        done = terminated or truncated
        last_info = info

    return ep_reward, steps, last_info


def evaluate(
    model_path: str,
    opponent: str,
    n_episodes: int,
    deterministic: bool,
    seed: Optional[int],
) -> EvalStats:
    opp = get_opponent(opponent)
    env = TicTacToeEnv(opponent_policy=opp, render_mode=None)

    if seed is not None:
        env.reset(seed=seed)

    model = MaskablePPO.load(model_path)

    stats = EvalStats()

    for _ in range(n_episodes):
        ep_reward, steps, info = run_episode(env, model, deterministic)

        stats.episodes += 1
        stats.total_reward += ep_reward
        stats.total_steps += steps

        if info.get("invalid_move"):
            stats.invalid += 1
            stats.losses += 1
        else:
            winner = info.get("winner", 0)
            if winner == 1:
                stats.wins += 1
            elif winner == -1:
                stats.losses += 1
            else:
                stats.draws += 1

    env.close()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/maskppo_tictactoe.zip")
    parser.add_argument("--opponent", choices=["random", "heuristic", "minimax"], default="random")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    stats = evaluate(
        model_path=args.model,
        opponent=args.opponent,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    eps = stats.episodes
    win_rate = stats.wins / eps if eps else 0.0
    loss_rate = stats.losses / eps if eps else 0.0
    draw_rate = stats.draws / eps if eps else 0.0
    invalid_rate = stats.invalid / eps if eps else 0.0
    avg_reward = stats.total_reward / eps if eps else 0.0
    avg_steps = stats.total_steps / eps if eps else 0.0

    print("\n=== Evaluation Results ===")
    print(f"Model:        {args.model}")
    print(f"Opponent:     {args.opponent}")
    print(f"Episodes:     {eps}")
    print(f"Deterministic:{args.deterministic}")
    print("")
    print(f"Wins:   {stats.wins:5d}  ({win_rate*100:6.2f}%)")
    print(f"Losses: {stats.losses:5d}  ({loss_rate*100:6.2f}%)")
    print(f"Draws:  {stats.draws:5d}  ({draw_rate*100:6.2f}%)")
    print(f"Invalid:{stats.invalid:5d}  ({invalid_rate*100:6.2f}%)")
    print("")
    print(f"Avg reward: {avg_reward:.4f}")
    print(f"Avg steps:  {avg_steps:.2f}")
    print("==========================\n")


if __name__ == "__main__":
    main()