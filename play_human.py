from __future__ import annotations

import argparse
import numpy as np

from tictactoe_env import TicTacToeEnv
from opponents import random_opponent, heuristic_opponent


def parse_move(s: str) -> int:
    """
    Accept:
      - 0..8 index
      - "row col" where row/col are 1..3
    """
    s = s.strip()
    if " " in s:
        r, c = s.split()
        r = int(r) - 1
        c = int(c) - 1
        return r * 3 + c
    return int(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    args = parser.parse_args()

    opp = random_opponent if args.opponent == "random" else heuristic_opponent
    env = TicTacToeEnv(opponent_policy=opp, render_mode="human")

    obs, info = env.reset()
    done = False

    print("You are X. Moves: enter 0-8, or 'row col' (1-3 each).")
    print("Index map:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")

    while not done:
        print("\nCurrent board:")
        env.render()

        move_str = input("Your move: ")
        try:
            action = parse_move(move_str)
        except Exception:
            print("Could not parse move.")
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info.get("invalid_move"):
            print("\nInvalid move. You lose.")
            break

        if done:
            print("\nFinal board:")
            env.render()
            if info.get("winner") == 1:
                print("You win!")
            elif info.get("winner") == -1:
                print("You lose.")
            else:
                print("Draw.")

    env.close()


if __name__ == "__main__":
    main()