from __future__ import annotations

import argparse
from stable_baselines3 import PPO

from tictactoe_env import TicTacToeEnv
from opponents import random_opponent, heuristic_opponent


def parse_move(s: str) -> int:
    s = s.strip()
    if " " in s:
        r, c = s.split()
        r = int(r) - 1
        c = int(c) - 1
        return r * 3 + c
    return int(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_tictactoe.zip")
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    args = parser.parse_args()

    opp = random_opponent if args.opponent == "random" else heuristic_opponent
    env = TicTacToeEnv(opponent_policy=opp, render_mode="human")
    model = PPO.load(args.model)

    obs, info = env.reset()
    done = False

    print("You are X, model also plays X when it's its turn? No: here YOU will choose whether you or the model plays X.")
    print("Choose mode:")
    print("  1) You (X) vs opponent (O)")
    print("  2) Model (X) vs opponent (O)")
    mode = input("Enter 1 or 2: ").strip()
    human_plays = (mode != "2")

    print("Index map:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")

    while not done:
        print("\nCurrent board:")
        print(env.render())

        if human_plays:
            move_str = input("Your move: ")
            try:
                action = parse_move(move_str)
            except Exception:
                print("Could not parse move.")
                continue
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            print(f"Model chooses: {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info.get("invalid_move"):
            print("\nInvalid move by X. Episode ends.")
            break

        if done:
            print("\nFinal board:")
            print(env.render())
            if info.get("winner") == 1:
                print("X wins!")
            elif info.get("winner") == -1:
                print("O wins!")
            else:
                print("Draw.")

    env.close()


if __name__ == "__main__":
    main()