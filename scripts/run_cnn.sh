#!/usr/bin/env bash
# Phase 1: CNN-DQN baseline. Usage: bash scripts/run_cnn.sh <run_name> [extra train.py args]
# Example: bash scripts/run_cnn.sh cnn_base --total_steps 60000
# Runs in the background with nohup, so it survives SSH disconnects.
set -e
RUN=${1:-cnn_base}; shift || true
source .venv/bin/activate
mkdir -p logs
nohup python train.py --model cnn --run_name "$RUN" "$@" > "logs/${RUN}_stdout.log" 2>&1 &
echo "started $RUN (pid $!). Watch with:  tail -f logs/${RUN}_stdout.log"
