#!/usr/bin/env bash
# One-time setup on Turing. Run from the repo root:  bash scripts/turing_setup.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-turing.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
python check_setup.py
