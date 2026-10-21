#!/usr/bin/env bash
# Final accuracy check (100 episodes) on the best checkpoint. Usage: bash scripts/final_eval.sh cnn_base
source .venv/bin/activate
python evaluate.py --ckpt checkpoints/${1:-cnn_base}_best.pt --episodes 100 --eps 0.0
