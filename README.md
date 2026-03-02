# TicTacToe RL (SB3 PPO)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python train_ppo.py --opponent mixed --timesteps 600000 --save-path models/ppo_tictactoe
```

## Evaluate model
```bash
python evaluate.py --model models/ppo_tictactoe.zip --opponent random --episodes 1000 --deterministic
python evaluate.py --model models/ppo_tictactoe.zip --opponent heuristic --episodes 1000 --deterministic
python evaluate.py --model models/ppo_tictactoe.zip --opponent minimax --episodes 1000 --deterministic
```

## Play vs the trained model
```bash
python play_vs_model.py --model models/ppo_tictactoe.zip --you O
```