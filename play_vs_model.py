from __future__ import annotations

import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from tictactoe_env import check_winner, is_draw


def render_board(board: np.ndarray) -> str:
    symbols = {1: "X", -1: "O", 0: " "}
    b = [symbols[int(v)] for v in board]
    rows = [
        f" {b[0]} | {b[1]} | {b[2]} ",
        f" {b[3]} | {b[4]} | {b[5]} ",
        f" {b[6]} | {b[7]} | {b[8]} ",
    ]
    sep = "---+---+---"
    return "\n".join([rows[0], sep, rows[1], sep, rows[2]])


def print_index_map() -> None:
    print("Index map:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")


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


def make_obs(board: np.ndarray, player_to_move: int) -> np.ndarray:
    """
    Must match the training env observation format:
      obs = [board(9), current_player(1)]
    where current_player is +1 for X-to-move, -1 for O-to-move.
    """
    return np.concatenate([board.astype(np.int8), np.array([player_to_move], dtype=np.int8)])


def masked_model_action(model: PPO, obs: np.ndarray, board: np.ndarray) -> int:
    """
    Choose the highest-probability LEGAL action according to the model.

    This prevents invalid moves during play, even if the model is imperfect.

    model: loaded SB3 PPO model with Discrete(9) actions
    obs: observation vector shape (10,)
    board: board vector shape (9,) with 0 empty, +1 X, -1 O
    """
    legal = np.flatnonzero(board == 0)
    if len(legal) == 0:
        return 0  # should not happen in normal play

    # Get the action distribution from the policy (PyTorch)
    obs_t = torch.as_tensor(obs).float().unsqueeze(0)  # (1, 10)
    dist = model.policy.get_distribution(obs_t)
    probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)  # (9,)

    # Mask illegal moves and pick the best legal move
    masked = np.zeros_like(probs)
    masked[legal] = probs[legal]

    if masked.sum() <= 0:
        # Fallback if distribution is degenerate
        return int(np.random.choice(legal))

    return int(masked.argmax())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_tictactoe.zip")
    parser.add_argument("--you", choices=["O", "X"], default="O",
                        help="Choose which side you play. Default: O (recommended).")
    parser.add_argument("--deterministic", action="store_true",
                        help="If set, model uses greedy legal argmax (default is already greedy via masking).")
    args = parser.parse_args()

    model = PPO.load(args.model)

    board = np.zeros(9, dtype=np.int8)

    # Decide sides
    # In our training setup, the model was trained as X (+1).
    # Playing as O is the most meaningful by default.
    human_mark = -1 if args.you.upper() == "O" else 1
    model_mark = -human_mark

    # X always goes first in tic-tac-toe
    player = 1  # +1 = X to move, -1 = O to move

    print(f"You are {'X' if human_mark == 1 else 'O'}. Model is {'X' if model_mark == 1 else 'O'}.")
    print("Moves: enter 0-8, or 'row col' (1-3 each).")
    print_index_map()

    while True:
        print("\nCurrent board:")
        print(render_board(board))

        if player == model_mark:
            obs = make_obs(board, player_to_move=player)
            action = masked_model_action(model, obs, board)
            print(f"Model plays: {action}")
            board[action] = model_mark
        else:
            move_str = input("Your move: ")
            try:
                action = parse_move(move_str)
            except Exception:
                print("Could not parse move.")
                continue

            if action < 0 or action > 8 or board[action] != 0:
                print("Illegal move. Try again.")
                continue

            board[action] = human_mark

        w = check_winner(board)
        if w == 1:
            print("\nFinal board:")
            print(render_board(board))
            print("X wins!")
            break
        if w == -1:
            print("\nFinal board:")
            print(render_board(board))
            print("O wins!")
            break
        if is_draw(board):
            print("\nFinal board:")
            print(render_board(board))
            print("Draw.")
            break

        player *= -1


if __name__ == "__main__":
    main()