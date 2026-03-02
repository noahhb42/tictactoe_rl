# TicTacToe RL (SB3 PPO)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Play (no training)
```bash
python play_human.py --opponent random
python play_human.py --opponent heuristic
```

## Train
```bash
python train_ppo.py --opponent random --timesteps 300000
# or
python train_ppo.py --opponent heuristic --timesteps 600000
```

## Play vs the trained model
```bash
python play_vs_model.py --model models/ppo_tictactoe.zip --you O
# optional:
python play_vs_model.py --model models/ppo_tictactoe.zip --you X
```