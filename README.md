# PeraMorphIQ: ViZDoom CNN-DQN to SNN-DQN

A reinforcement-learning agent that plays ViZDoom from raw pixels. The final goal is a **spiking neural network (SNN)** agent that can later be mapped onto an **FPGA neuromorphic accelerator**.

The work is split into two phases (from the instructor meeting, 21 Oct 2026):

1. **Phase 1: CNN-DQN.** Train an ordinary CNN agent on the ViZDoom `basic` scenario until **success rate > 90%** and **mean reward > 90**.
2. **Phase 2: SNN-DQN.** Only after Phase 1 passes. Reuse the same pipeline and compare against the CNN.

**Phase 1 is complete: 100% success, mean reward 95.00** (see [Results](#results)).

---

## Table of contents

1. [Task and metrics](#task-and-metrics)
2. [Results](#results)
3. [Key finding: why the old agent was stuck at ~80](#key-finding-why-the-old-agent-was-stuck-at-80)
4. [Delta-aiming action space](#delta-aiming-action-space)
5. [Repository layout](#repository-layout)
6. [Environment and compute](#environment-and-compute)
7. [How to reproduce](#how-to-reproduce)
8. [Configuration reference](#configuration-reference)
9. [Models](#models)
10. [Training and evaluation protocol](#training-and-evaluation-protocol)
11. [Phase 2: SNN plan](#phase-2-snn-plan)
12. [Limitations and unverified assumptions](#limitations-and-unverified-assumptions)
13. [Git history and branches](#git-history-and-branches)
14. [Security notes](#security-notes)

---

## Task and metrics

**Scenario:** ViZDoom `basic`. One stationary monster appears at a random horizontal position at the far end of a room. The agent must line up the crosshair and shoot it.

| Item | Value |
|---|---|
| Observation | grayscale 160x120 frame, resized to `img_size` x `img_size`, 4 frames stacked |
| Reward (unchanged, raw) | -1 per tic, about +101 net for the kill, about -5 per missed shot |
| Episode timeout | 300 tics |
| Success | episode reward > 0 (monster killed) |
| Mean reward | average raw episode reward (not the 0.01-scaled value used for learning) |

The reward function, scenario file, and termination conditions are **never modified**. Only the action space differs between experiments.

---

## Results

### Summary

| Run | Action space | frame_skip | Mean reward | Success | Evaluation |
|---|---|---|---|---|---|
| `cnn_base` / `cnn_fast` | strafe L/R + attack (3) | 4 | ~77-81 | 100% | best-of-noisy periodic evals; 200-ep re-eval of `cnn_base` best: 80.5 |
| `cnn_tuned` (3 runs) | strafe (3) | 4 | 83.4 / 83.9 / 83.6 (best) | 100% | selection bias: best of ~50 noisy 30-ep evals; honest value ~80-81 |
| `cnn_fs2` | strafe (3) | 2 | 80.3 (200 ep) | 100% | 300k steps, cosine LR decay |
| `cnn_combo` | strafe + move-and-attack (5) | 2 | 78.5 (200 ep) | 100% | worse: combos were rarely chosen and cost extra misses |
| **`cnn_aim_delta`** | **delta turn + attack (17)** | **1** | **95.00** | **100%** | **200 unseen held-out episodes** |

### Final result: `cnn_aim_delta`

| Metric | Value |
|---|---|
| Mean reward (200 held-out episodes, never used for training or checkpoint selection) | **95.00** |
| Standard deviation | 0.0 (every episode scored 95) |
| Success rate | **100%** |
| Best-checkpoint re-eval on the 200-episode selection set | 95.0 (step 260000) |
| Oracle best-case ceiling, same actions, 100 fixed seeds | 94.92 (std 0.27, median 95, mean 6.08 tics) |
| Oracle with `aim_unit 2.0` | 94.77 |
| Old strafe best-case ceiling | ~84 |

Both Phase 1 targets (success > 90%, mean reward > 90) are met. The agent is at the oracle ceiling, so there is no further headroom to gain under this reward function.

Training curve highlights (300k steps, `eval_every` 10k, 50 episodes per eval):

- 30k steps: 88.2, 100% success. 40k: 94.3.
- 60k-80k: a transient collapse (89.8, 30.4, 90.3) right after exploration reached its floor. This is typical DQN instability; it recovered by 90k.
- From about 130k onward: 92 to 95, 100% success at every evaluation.

Result files are in `results/cnn_aim_delta/` (`*_evals.csv`, `*_episodes.csv`, `*_final.csv`, `*_holdout.csv`, `*_holdout_episodes.csv`, `*_curve.png`).

---

## Key finding: why the old agent was stuck at ~80

Evidence, all from existing logs and the oracle search:

1. **Rewards come in steps of 4** (95, 91, 87, 83, ...) with `frame_skip 4`, because each decision costs `frame_skip` tics at -1 per tic. 95 was the best single episode ever seen, and it only happens when the monster spawns near the screen centre (about 15% of episodes).
2. The monster spawns about +/-40 px from the screen centre. With strafing, **time spent sliding into line is a fixed cost**, so the mean over spawns is capped.
3. A best-case search over strafe actions (same spawns; strafe k tics, coast j tics, then fire) gave **84.0 / 83.5 / 82.8** for `frame_skip` 1 / 2 / 4. The agent at 80-81 was therefore already about 3 points under a hard ceiling.
4. Training changes could not fix this: a hand-written aim-and-shoot script scored only 78-79, and combined move-and-shoot actions made things worse.
5. The "loss ~0.000" seen in the logs is mostly a scale artefact: rewards are multiplied by 0.01, so Q-values are around 1 and Huber loss prints as `0.000` even while learning. The flat curve was the ceiling, not a stalled optimiser.

Conclusion: reaching 90+ required changing how the agent aims, not tuning the learner.

---

## Delta-aiming action space

Strafing is replaced by instantaneous turning, so the agent can line up the monster in one or two decisions instead of many tics.

**Buttons** (set in `env.py` after loading `basic.cfg`, so reward and termination stay untouched): `TURN_LEFT_RIGHT_DELTA`, `ATTACK`.

**Actions (17):**

| Index | Meaning |
|---|---|
| 0-7 | turn only, signed deltas -20, -8, -3, -1, +1, +3, +8, +20 |
| 8 | ATTACK only |
| 9-16 | the same 8 signed turns **plus ATTACK in the same tic** (`--combo_actions 1`) |

Other design points:

- **`frame_skip` handling.** The delta is applied on the first tic of a decision only. The remaining `frame_skip - 1` tics repeat just the ATTACK component, so `frame_skip` never multiplies the turn angle. The final run uses `frame_skip 1`.
- **`aim_unit`.** A multiplier on the deltas, because the unit of `TURN_LEFT_RIGHT_DELTA` was not verified from documentation. The oracle was run at 1.0 and 2.0 and both reach about 94.8-94.9, so 1.0 is used.
- **Other modes kept for comparison:** `aim_mode=strafe` (original baseline, default in `config.py`) and `aim_mode=turn` (digital `TURN_LEFT`, `TURN_RIGHT`, `ATTACK`; implemented but not evaluated).
- `evaluate.py` and `train.py` rebuild the environment from the same config (`make_env`), so training and evaluation always use an identical action mapping. The aim settings are stored inside each checkpoint.

### Oracle feasibility search (`oracle_aim.py`)

Before spending GPU time, a CPU-only best-case search was run: for each of 100 fixed seeds it tries many open-loop programs (up to 2 delta turns, then fire, or a turn+attack combo) and keeps the best per seed. This is an **upper bound with hindsight, not a learnable policy**. It showed a 94.9 ceiling versus ~84 for strafing, which justified training.

---

## Repository layout

| Path | Purpose |
|---|---|
| `config.py` | All hyperparameters (dataclass `Config`). Every field is also a `train.py` command-line flag. |
| `env.py` | ViZDoom wrapper: grayscale, resize, frame skip, frame stack, discrete actions, `aim_mode` handling |
| `model.py` | `CnnDQN` (ReLU baseline) and `SpikingDQN` (LIF, fixed beta). `build_model(cfg, n_actions)` picks one. |
| `replay.py` | Replay buffer |
| `train.py` | DQN training, periodic eval, checkpoints, top-k re-scoring, held-out evaluation |
| `evaluate.py` | Evaluate a saved checkpoint (greedy or with epsilon, optional video) |
| `oracle_aim.py` | CPU best-case feasibility search for the aiming action space |
| `check_setup.py` | Sanity check: GPU, imports, one random episode, SNN forward/backward speed and VRAM |
| `requirements.txt` | Common Python dependencies (includes `vizdoom==1.3.2`) |
| `requirements-turing.txt` | CUDA 12.1 PyTorch build for the Turing server |
| `requirements-cpu.txt` | CPU PyTorch build |
| `scripts/turing_setup.sh` | One-time venv and dependency setup on Turing |
| `scripts/run_cnn.sh` | Launch a CNN training run in the background with `nohup` |
| `scripts/final_eval.sh` | 100-episode evaluation of a run's best checkpoint |
| `results/` | Committed results per run (CSVs and curves; no checkpoints) |

`results/archive_snn_v1_v2/` holds the earlier unstable SNN runs, kept for reference.

---

## Environment and compute

Experiments were run on two environments.

### Turing server (GPU, SSH)

Used for long-running training that must survive disconnects.

```bash
git clone https://github.com/ayyash15/vizdoom.git
cd vizdoom
bash scripts/turing_setup.sh                 # creates .venv, installs requirements-turing.txt, runs check_setup.py
bash scripts/run_cnn.sh <run_name> [extra train.py args]
tail -f logs/<run_name>_stdout.log           # watch progress (training runs under nohup)
bash scripts/final_eval.sh <run_name>        # 100-episode check of checkpoints/<run_name>_best.pt
```

Example: `bash scripts/run_cnn.sh cnn_base --total_steps 60000`.

### Google Colab (T4 GPU)

Used for the delta-aiming oracle and training run (`cnn_aim_delta`: about 18 minutes at roughly 280-350 it/s).

```python
from google.colab import userdata
tok = userdata.get("GH_TOKEN")                # token stored in Colab Secrets, never typed into the notebook
%cd /content
!git clone https://{tok}@github.com/ayyash15/vizdoom.git
%cd /content/vizdoom
!pip install -q vizdoom==1.3.2 snntorch==1.0.0 gymnasium==1.4.0 opencv-python-headless numpy matplotlib pandas scipy tqdm tensorboard
import torch; print("CUDA:", torch.cuda.is_available())     # confirm True before training
```

Colab notes:

- Do **not** `pip install -r requirements*.txt` on Colab. It would replace Colab's preinstalled PyTorch. Install the packages above instead.
- Use `%cd` (not `!cd`) so the directory change persists.
- To push results from Colab, use a GitHub token read from Colab Secrets, commit only `results/<run>/`, and reset the remote URL afterwards (see [Security notes](#security-notes)).

---

## How to reproduce

### 1. Feasibility check (CPU only, no training)

```bash
python oracle_aim.py --aim_mode delta --combo 1 --seeds 100 --aim_unit 1.0
python oracle_aim.py --aim_mode delta --combo 1 --seeds 100 --aim_unit 2.0
```

Proceed only if the printed `mean` is above 90. Reference output: 94.92 and 94.77.

### 2. Train the delta-aiming CNN-DQN (the `cnn_aim_delta` run)

```bash
python train.py --model cnn --run_name cnn_aim_delta \
  --aim_mode delta --combo_actions 1 --aim_unit 1.0 \
  --frame_skip 1 --total_steps 300000 --eps_decay_frac 0.2 --img_size 84
```

Check the first printed lines: you should see `aim_mode: delta`, 17 actions, and the 17 action vectors. At the end, look for the `[holdout]` line.

### 3. Evaluate a checkpoint

```bash
python evaluate.py --ckpt checkpoints/cnn_aim_delta_best.pt --episodes 200
```

### 4. Reproduce the old strafe baseline

```bash
python train.py --model cnn --run_name cnn_fs2 --frame_skip 2 --total_steps 300000
```

(`config.py` defaults are the old strafe baseline: `aim_mode=strafe`, `frame_skip=2`.)

### 5. Outputs written by training

| File | Content |
|---|---|
| `logs/<run>_episodes.csv` | step, episode reward, epsilon for every training episode |
| `logs/<run>_evals.csv` | periodic eval: step, mean reward, success rate |
| `logs/<run>_final.csv` | top-k checkpoints re-scored on identical episodes |
| `logs/<run>_holdout.csv` | held-out summary: episodes, mean, std, 95% CI, median, success |
| `logs/<run>_holdout_episodes.csv` | the individual held-out episode rewards |
| `logs/<run>_curve.png` | training and evaluation curves |
| `checkpoints/<run>_best.pt` | selected checkpoint (not committed) |

To publish results, copy `logs/<run>_*.csv` and `logs/<run>_curve.png` into `results/<run>/` and commit only that folder.

---

## Configuration reference

All fields live in `config.py` and can be overridden as `--field value`. Defaults below describe the **baseline** configuration; the delta-aiming run overrides the flags shown above.

| Group | Field | Default | Notes |
|---|---|---|---|
| Environment | `scenario` | `basic` | |
| | `frame_skip` | 2 | decision costs this many tics; delta run uses 1 |
| | `img_size` | 64 | delta run uses 84 so the monster offset is visible at finer resolution |
| | `stack` | 4 | stacked frames |
| | `reward_scale` | 0.01 | raw rewards scaled for stable learning; evaluation reports raw rewards |
| | `combo_actions` | 0 | 1 adds move+attack (strafe) or turn+attack (delta) actions |
| Aiming | `aim_mode` | `strafe` | `strafe` / `turn` / `delta` |
| | `aim_deltas` | `1,3,8,20` | delta magnitudes, applied as +/- |
| | `aim_unit` | 1.0 | multiplier on deltas |
| Model | `model` | `cnn` | `cnn` / `snn` |
| | `rate_reg` | 0.0 | firing-rate penalty weight (SNN only) |
| SNN | `T`, `beta`, `hidden` | 8, 0.9, 256 | timesteps per decision, fixed LIF decay, FC width |
| DQN | `total_steps` | 300000 | agent decisions |
| | `lr`, `lr_decay`, `lr_final_frac` | 5e-5, 1, 0.1 | cosine decay to `lr * lr_final_frac` |
| | `batch_size`, `gamma` | 32, 0.99 | |
| | `buffer_size`, `warmup` | 30000, 1000 | roughly 600 MB RAM at 64x64 |
| | `train_every`, `target_update` | 2, 1000 | |
| | `double_dqn` | 1 | Double DQN |
| | `eps_start`, `eps_end`, `eps_decay_frac` | 1.0, 0.01, 0.4 | delta run uses `eps_decay_frac 0.2` |
| | `grad_clip` | 10.0 | |
| Evaluation | `eval_every`, `eval_episodes`, `eval_eps` | 10000, 50, 0.0 | greedy periodic evaluation |
| | `final_top_k`, `final_eval_episodes` | 3, 200 | top checkpoints re-scored on identical spawns |
| Bookkeeping | `run_name`, `seed` | `cnn_fs2`, 42 | |

---

## Models

Both networks share the same layer sizes (Conv 16 (5x5, stride 2) -> Conv 32 (3x3, stride 2) -> Conv 64 (3x3, stride 2) -> FC 256 -> FC n_actions):

- **`CnnDQN`**: ReLU activations, one forward pass. The Phase 1 baseline.
- **`SpikingDQN`**: Conv-LIF x3, FC-LIF, then a non-spiking LIF readout whose membrane potential gives the Q-values. Design choices for FPGA friendliness: constant-current input encoding (first convolution computed once and injected every timestep), only conv, linear and LIF layers (no batch norm, no max-pool), and a **fixed** `beta` (learnable beta drifted past 1.0 and destabilised training; a fixed beta is also what hardware will implement). Surrogate gradient: fast sigmoid (slope 25). Spike rates per layer are logged by `evaluate.py`.

---

## Training and evaluation protocol

- Standard DQN with experience replay, a target network, Double DQN, Huber loss and gradient clipping. Linear epsilon decay, cosine learning-rate decay.
- Periodic evaluation every `eval_every` steps is greedy, on a separate environment seed (`seed + 1`).
- At the end, the top `final_top_k` checkpoints (by periodic score, restricted to >= 95% success where possible) are re-scored on **200 identical episodes** (`seed + 2`), and the winner is saved as `<run>_best.pt`.
- Because that selection set is also used for the choice, the winner is **re-scored once more on 200 unseen episodes** (`seed + 3`). The `*_holdout.csv` value is the number to report. It includes mean, std, 95% CI, median and success.
- Do not quote the best of many small periodic evaluations as a final score: that was the source of the inflated 83.x numbers earlier.

---

## Phase 2: SNN plan

1. Train the SNN with the **same action space and flags** as `cnn_aim_delta`, so the comparison is fair: `--model snn --aim_mode delta --combo_actions 1 --frame_skip 1 --img_size 84`.
2. Start with a short test run: with T=8 timesteps it is several times slower than the CNN.
3. Suggested stabilisers, since SNN training is usually less stable: lower learning rate (e.g. `--lr 2e-5`), `--rate_reg 0.01` to `0.05`, `--double_dqn 1`, and keep the top-k checkpoint selection.
4. Compare against the CNN baseline: **mean reward 95.00, success 100%**.
5. Later: int8 quantisation with fixed beta and threshold, spike-sparsity logging, and mapping onto the FPGA accelerator.

---

## Limitations and unverified assumptions

- **The action space changed.** The 95.00 result uses delta turning instead of strafing. The reward function and termination are identical, but the instructor should be told, because the earlier analysis called for approval before changing the action space.
- **The score is at the oracle ceiling.** All 200 held-out episodes scored 95, so the standard deviation is 0 and the confidence interval is degenerate.
- **Units of `TURN_LEFT_RIGHT_DELTA`** were not verified from documentation. They were checked only indirectly through the oracle at `aim_unit` 1.0 and 2.0.
- **Turn-and-fire in one tic** (the turn+ATTACK combos) was not verified independently; the 94.9 oracle shows that some program using these actions kills in about 6 tics.
- `aim_mode=turn` is implemented but not evaluated.
- The oracle uses a hindsight search over a limited family of programs and 100 seeds; it is an upper bound, not a policy.
- The run used a single training seed (42); stability across seeds is untested. A transient collapse was observed at 60k-80k steps before recovery.
- `config.py` defaults still describe the old strafe baseline. Always pass the flags in [How to reproduce](#how-to-reproduce).
- Checkpoints are not stored in git; only code and results are committed.

---

## Git history and branches

- `main` contains the delta-aiming code, the oracle, the held-out evaluation and all results.
- The original baseline work is preserved under the git tag **`cnn-baseline-final`** (the former `cnn-baseline` branch).
- The delta-aiming work was developed on a `cnn-aim` branch and merged into `main`.

---

## Security notes

- Never paste a GitHub token into a notebook cell or chat. Read it from Colab Secrets (`userdata.get("GH_TOKEN")`).
- After pushing from Colab, remove the token from the stored remote: `git remote set-url origin https://github.com/ayyash15/vizdoom.git`.
- Clear notebook outputs before sharing a notebook, and revoke any token that has been exposed (GitHub -> Settings -> Developer settings).
- Do not commit checkpoints, `logs/`, or credentials.
