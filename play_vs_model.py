from __future__ import annotations

import argparse
import numpy as np
from sb3_contrib import MaskablePPO

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
    s = s.strip()
    if " " in s:
        r, c = s.split()
        r = int(r) - 1
        c = int(c) - 1
        return r * 3 + c
    return int(s)


def make_obs(board: np.ndarray, player_to_move: int) -> np.ndarray:
    return np.concatenate([board.astype(np.int8), np.array([player_to_move], dtype=np.int8)])


def action_mask_from_board(board: np.ndarray) -> np.ndarray:
    # True for legal actions
    return (board == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/maskppo_tictactoe.zip")
    parser.add_argument("--you", choices=["O", "X"], default="O")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model)

    board = np.zeros(9, dtype=np.int8)

    human_mark = -1 if args.you.upper() == "O" else 1
    model_mark = -human_mark

    # X always goes first
    player = 1

    print(f"You are {'X' if human_mark == 1 else 'O'}. Model is {'X' if model_mark == 1 else 'O'}.")
    print("Moves: enter 0-8, or 'row col' (1-3 each).")
    print_index_map()

    while True:
        print("\nCurrent board:")
        print(render_board(board))

        if player == model_mark:
            obs = make_obs(board, player_to_move=player)
            mask = action_mask_from_board(board)
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)
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