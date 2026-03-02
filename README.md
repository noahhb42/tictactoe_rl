## Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Play vs baseline (no training)
python play_human.py --opponent random
python play_human.py --opponent heuristic

## Train PPO
python train_ppo.py --opponent random --timesteps 300000
# model saved to models/ppo_tictactoe(.zip)

## Play against trained model (or watch it play)
python play_model.py --model models/ppo_tictactoe.zip --opponent heuristic