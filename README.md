# PeraMorphIQ

ViZDoom agent project. Final goal: a spiking neural network (SNN) agent for an FPGA accelerator.

## Plan (from instructor meeting, 21 Oct 2026)
1. **Phase 1: CNN-DQN.** Train a normal CNN agent on ViZDoom `basic` until success rate is **above 90%**.
2. **Phase 2: SNN-DQN.** Only after Phase 1 passes. Reuse the same pipeline and compare against the CNN.

Success = episode reward > 0 (monster killed). "Accuracy" in this project means success rate over 100 evaluation episodes.

## Status
- [x] Pipeline works (env, replay, DQN, eval) and SNN model exists
- [x] CNN-DQN above 90% success on `basic` (Phase 1 done, see results below)
- [ ] SNN-DQN  <- current task
- Old SNN results (v2 best 77%, final 47%) are in `results/archive_snn_v1_v2/`

## Results

### Phase 1: CNN-DQN baseline (`cnn_base`)
| Item | Value |
|---|---|
| Success rate | **100%** (re-eval of best checkpoint, 200 episodes) |
| Mean reward | 80.5 |
| Best checkpoint | step 300000 |
| Target | above 90% |

Result files: `results/cnn_base/` (`cnn_base_evals.csv`, `cnn_base_episodes.csv`, `cnn_base_curve.png`).
The checkpoint `cnn_base_best.pt` is not in git. Ask the team for a copy.

This is the number the SNN has to be compared against in Phase 2.

## Files
| File | Purpose |
|---|---|
| `config.py` | All hyperparameters (default model = `cnn`) |
| `env.py` | ViZDoom wrapper: grayscale, 64x64, 4-frame stack, frame skip 4 |
| `model.py` | `CnnDQN` and `SpikingDQN` |
| `replay.py` | Replay buffer |
| `train.py` | DQN training, evals, checkpoints |
| `evaluate.py` | Evaluate a checkpoint |
| `check_setup.py` | Environment sanity check |
| `scripts/` | Turing setup and run scripts |

## Run on Turing (RTX 3090 Ti, Ubuntu 22.04, no sudo)
```bash
ssh <your-eNNNNN>@turing.ce.pdn.ac.lk
git clone https://github.com/ayyash15/vizdoom.git && cd vizdoom
bash scripts/turing_setup.sh        # one time: venv + packages + check
nvidia-smi                          # server is shared: check GPU use first
bash scripts/run_cnn.sh cnn_base    # trains in background
tail -f logs/cnn_base_stdout.log
bash scripts/final_eval.sh cnn_base # 100-episode accuracy
```
Evals print as `[eval] step N: ... success X%` every 5,000 steps. Best checkpoint: `checkpoints/cnn_base_best.pt`.

Tips:
- Different settings = different `--run_name`, so runs do not overwrite each other. Example: `bash scripts/run_cnn.sh cnn_lr1e4 --lr 1e-4 --batch_size 64`
- Use `CUDA_VISIBLE_DEVICES=0` if the server has more than one GPU.
- Do not commit `checkpoints/` or `logs/`. Copy result CSVs and plots you want to keep into `results/`.

## Getting results off Turing
You cannot push to GitHub from Turing. Copy files to your PC, then push from there.

Run these in PowerShell on your PC (not inside the SSH session):
```powershell
cd <your local repo folder>
scp <your-eNNNNN>@turing.ce.pdn.ac.lk:~/vizdoom/results/<run_name>/* results/<run_name>/
scp <your-eNNNNN>@turing.ce.pdn.ac.lk:~/vizdoom/checkpoints/<run_name>_best.pt .
git checkout cnn-baseline
git add results/<run_name>      # never use "git add ." (it would add .pt files)
git commit -m "your message"
git push
```

## Git rules
- Work on branch `cnn-baseline`. Merge to `main` when the team agrees (CNN has passed 90%).
- Commit code, configs and small result files only.
- Do not commit `.pt` checkpoints or `logs/`.
