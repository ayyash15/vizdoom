# PeraMorphIQ — Spiking Neural Network DOOM Agent

A hardware-oriented Spiking Neural Network (SNN) agent that learns to play DOOM autonomously using [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) and Deep Q-Learning.

**PeraMorphIQ** is the software stage of a project aimed at deploying a spiking neural network on an FPGA-based neuromorphic accelerator. The network is designed around hardware-friendly building blocks, including convolutional layers, linear layers, and leaky integrate-and-fire (LIF) neurons.

> **Current status (20 October 2026):** The baseline SNN-DQN pipeline has been implemented and tested on the ViZDoom `basic` scenario. The v2 training run has completed, but evaluation revealed instability. Diagnostic experiments and ANN/CNN baseline comparisons are the next priority. FPGA export has not started.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Experimental Results](#experimental-results)
- [Baseline Diagnosis](#baseline-diagnosis)
- [Version 2 Improvements](#version-2-improvements)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Ablations](#experimental-ablations)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Technologies](#technologies)

---

## Project Overview

The project investigates whether an SNN can learn effective control policies in a visual reinforcement-learning environment while maintaining characteristics suitable for future neuromorphic hardware deployment.

The agent receives visual observations from DOOM, processes them through a neural network, and selects actions without human input.

The primary objectives are:

- Train an SNN-DQN agent to play the ViZDoom `basic` scenario.
- Investigate training stability and policy performance.
- Compare spiking and non-spiking neural network baselines.
- Measure spike activity as a proxy for potential neuromorphic hardware efficiency.
- Study the trade-off between temporal simulation steps, performance, and latency.
- Prepare the network for eventual FPGA implementation.

**Important:** Low spike activity is a potential indicator of reduced neuromorphic computation, but it does not directly establish energy efficiency. Actual hardware energy measurements will be needed to validate that claim.

## System Architecture

The project follows a reinforcement-learning pipeline:

1. **Environment:** ViZDoom generates visual observations and rewards.
2. **Preprocessing:** Frames are converted to grayscale, resized, and stacked.
3. **Neural network:** The SNN processes observations and estimates action values.
4. **Action selection:** An epsilon-greedy policy balances exploration and exploitation during training.
5. **Learning:** DQN uses experience replay and a target network to learn from transitions.
6. **Evaluation:** The agent is evaluated without human input, using success rate, reward, action distribution, and spike activity.

### Network Architecture

The current SNN architecture is:

`Input → Conv-LIF × 3 → FC-LIF → LIF Readout → Q-values`

The network uses:

- Three convolutional spiking layers.
- A fully connected spiking layer.
- A non-spiking-output LIF readout whose membrane potential is used to estimate Q-values.
- Direct, constant-current input encoding.
- Surrogate gradients through [snnTorch](https://snntorch.readthedocs.io/).

The architecture avoids batch normalization and max-pooling to simplify the path toward hardware implementation.

The temporal simulation length \(T\) controls how many spiking timesteps are used to process an observation.

## Methodology

### Environment and Observations

| Parameter | Configuration |
|---|---|
| Environment | ViZDoom |
| Initial scenario | `basic` |
| Original frame resolution | Environment-provided observation |
| Input color format | Grayscale |
| Processed frame size | 64 × 64 |
| Frame stacking | 4 frames |
| Frame skip | 4 |
| Agent input | Stacked visual observations |
| Control | Autonomous; no human input |

### Reinforcement Learning

The training pipeline uses Deep Q-Network learning with:

- Experience replay.
- Epsilon-greedy exploration.
- A target network.
- Double DQN in v2.
- Huber loss.
- Gradient clipping.
- Reward scaling.

The v2 configuration uses a lower learning rate of \(5 \times 10^{-5}\) and fixed, non-learnable LIF decay parameters (`beta`) to improve training stability and align with the intended hardware design.

### Evaluation Metrics

The agent is evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Success rate | Fraction of episodes meeting the success criterion |
| Mean episode reward | Measures overall task performance |
| Episode outcomes | Identifies successful and failed runs |
| Action distribution | Detects repetitive or potentially ineffective behavior |
| Per-layer spike rate | Measures neural activity as a hardware-efficiency proxy |
| Training stability | Tracks performance across checkpoints |
| Inference latency | Planned measurement for temporal-step comparisons |

Success is currently defined as an episode reward greater than zero. This criterion should be kept consistent across model comparisons.

---

## Experimental Results

### Baseline SNN-DQN — v1

The initial baseline was trained for approximately 50,000 steps, including resumed training.

| Metric | Observed result |
|---|---|
| Scenario | `basic` |
| Training budget | 50,000 steps, resumed |
| Mean reward, best checkpoint | -76.5 |
| Mean reward, last checkpoint | -67.9 |
| Approximate success rate | 55–60% |
| Training stability | Unstable |
| Reward pattern | Large oscillations between successful and failed episodes |

Episode outcomes were approximately bimodal:

- Successful episodes: around +95 reward.
- Failed or timed-out episodes: around -410 reward.

These results indicate that average reward alone can obscure the agent's behavior. Success rate and the distribution of episode outcomes are therefore important complementary metrics.

### Baseline Spike Activity

Measured spike rates for the original SNN:

| Layer | Spikes per neuron per timestep |
|---|---:|
| L1 | 0.25 |
| L2 | 0.40 |
| L3 | 0.58 |
| L4 | 0.61 |

Activity increases in the deeper layers, suggesting that firing-rate control may be worth investigating.

These measurements are preliminary. The exact firing-rate definition, measurement procedure, and relationship to hardware energy must be documented before drawing quantitative efficiency conclusions.

### Version 2 — SNN-DQN

The v2 training run completed 50,000 steps. Evaluation revealed a substantial decline after the best checkpoint.

| Training step | Evaluation success rate |
|---:|---:|
| 5,000 | Approximately 60% |
| 15,000 | 77% — best observed checkpoint |
| 50,000 | 47% — final checkpoint |

The 5,000-step figure is approximate. The reported best checkpoint was selected by success rate, with success defined as reward greater than zero.

**Current finding:** Training longer did not produce consistently better performance. The decline suggests instability, although evaluation variance and implementation details must also be considered.

A simple-policy baseline is needed to determine how much of the observed success rate reflects meaningful learning. Any baseline success-rate estimate must be measured experimentally before being reported as a result.

---

## Baseline Diagnosis

The following hypotheses are being investigated:

1. **Learning instability:** The agent's performance improves initially but degrades during continued training.
2. **Action repetition:** Greedy evaluation may produce repetitive behavior that leads to episode failure.
3. **Weak action-value separation:** Q-values may not clearly distinguish useful actions from poor ones.
4. **Spiking dynamics:** LIF state, firing rates, and surrogate-gradient behavior may affect learning.
5. **Shared DQN issues:** Exploration scheduling, replay sampling, reward scaling, target-network updates, or other training settings may contribute to instability.

The v2 configuration fixes the LIF decay parameter, so learnable decay drift is no longer the primary explanation for that run. Other SNN-specific and DQN-related causes remain possible.

### Diagnostic Experiments

#### 1. Inspect successful and failed episodes

Run the diagnostic script against the best v2 checkpoint:

```powershell
python diagnose.py --ckpt checkpoints\v2_best.pt --episodes 30
```

The script is intended to compare action selection and Q-value distributions across successful and failed episodes.

#### 2. Train a non-spiking baseline

If the updated training script supports the ANN flag:

```powershell
python train.py --scenario basic --total_steps 30000 --ann 1 --run_name ann
```

The goal is to compare a conventional neural network with the SNN under equivalent conditions.

If the ANN performs well while the SNN remains unstable, the spiking implementation becomes a stronger candidate for investigation. If both models struggle, the shared learning setup or environment may be responsible.

The result is diagnostic evidence rather than definitive proof; architecture, parameter count, preprocessing, and training conditions must be controlled.

---

## Version 2 Improvements

The v2 training pipeline introduced the following changes:

- Fixed LIF decay parameters instead of learning `beta`.
- Double DQN for action-value learning.
- Reduced learning rate of \(5 \times 10^{-5}\).
- Evaluation every 5,000 training steps.
- Evaluation across 30 episodes.
- Small evaluation epsilon of 0.02.
- Best-checkpoint selection using success rate.
- Checkpoint saving at each evaluation.
- Run-name tags to prevent different experiments from overwriting one another.

These changes improve experimental control and reproducibility, but the observed results show that they have not yet resolved the training instability.

---

## Installation

### Requirements

- Windows 10/11.
- Python 3.11 or 3.12.
- ViZDoom and Gymnasium.
- PyTorch.
- snnTorch.
- An NVIDIA GPU for accelerated training, if available.

The original development environment was tested with:

- PyTorch 2.6.0, CUDA 12.4 build.
- snnTorch 1.0.0.
- ViZDoom 1.3.2.
- Gymnasium 1.4.0.
- NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

Other configurations may work but have not necessarily been tested.

### 1. Clone the repository

```powershell
git clone <YOUR_REPOSITORY_URL>
cd doom_snn
```

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL.

### 2. Create and activate a virtual environment

```powershell
python -m venv doom_snn_env
.\doom_snn_env\Scripts\Activate.ps1
```

### 3. Install dependencies

For an NVIDIA GPU with a compatible CUDA setup:

```powershell
pip install -r requirements-gpu.txt
```

For CPU-only execution:

```powershell
pip install -r requirements-cpu.txt
```

If required by your dependency files, install the general project requirements:

```powershell
pip install -r requirements.txt
```

Check the installation:

```powershell
python check_setup.py
```

**Note:** The dependency files and installation commands should be verified against the current repository to ensure that packages are not installed redundantly or incompatibly.

---

## Usage

Run commands from the project directory with the virtual environment activated.

### Train the SNN-DQN agent

```powershell
python train.py --scenario basic --total_steps 50000 --run_name v2
```

### Evaluate a checkpoint

```powershell
python evaluate.py --ckpt checkpoints\v2_best.pt --episodes 30
```

### Render an evaluation episode and record video

```powershell
python evaluate.py --ckpt checkpoints\v2_best.pt --episodes 3 --render --video
```

### Inspect failure behavior

```powershell
python diagnose.py --ckpt checkpoints\v2_best.pt --episodes 30
```

The diagnostic command requires the updated `diagnose.py` script to be present in the repository.

### Train the CNN-DQN baseline

If the updated training script supports the `--model` option:

```powershell
python train.py --model cnn --run_name cnn_base
```

### Train the SNN with firing-rate regularization

If the updated training script supports the `--rate_reg` option:

```powershell
python train.py --rate_reg 0.05 --run_name snn_reg05
```

### Test a different temporal simulation length

```powershell
python train.py --T 4 --run_name v2_T4
```

These optional commands depend on the relevant flags being implemented in the current version of `train.py`. Confirm the available arguments with:

```powershell
python train.py --help
```

### Output files

Typical outputs include:

| Path | Purpose |
|---|---|
| `logs/<run>_episodes.csv` | Per-episode records |
| `logs/<run>_evals.csv` | Evaluation history |
| `logs/<run>_curve.png` | Training/evaluation plot |
| `checkpoints/<run>_best.pt` | Best saved checkpoint |
| `videos/` | Optional evaluation recordings |

Actual filenames depend on the logging implementation. Checkpoints, videos, and large generated outputs should generally be excluded from Git unless there is a specific reason to version them.

---

## Experimental Ablations

The next experiments will examine the relationship between task performance, network activity, and computational cost.

### 1. ANN/CNN versus SNN

Compare a conventional neural network baseline with the SNN-DQN.

**Objective:** Determine whether the observed instability is specific to spiking dynamics or also occurs with conventional neural networks.

### 2. Temporal simulation length

Evaluate:

- \(T = 4\)
- \(T = 8\)
- \(T = 16\)

**Objective:** Measure the trade-off between success rate, reward, spike activity, and inference latency.

Keep the training budget and other settings comparable. Ideally, repeat each configuration with multiple random seeds.

### 3. Firing-rate regularization

Investigate an SNN training objective of the form:

\[
L_{\text{total}} = L_{\text{DQN}} + \lambda L_{\text{rate}}
\]

where:

- \(L_{\text{DQN}}\) is the reinforcement-learning loss.
- \(L_{\text{rate}}\) penalizes excessive spiking activity.
- \(\lambda\) controls the strength of regularization.

The implementation of \(L_{\text{rate}}\) must be documented, including whether it penalizes average firing rate, layer-specific activity, or deviation from a target rate.

**Objective:** Reduce spike activity while preserving task performance.

### 4. Harder environment

After establishing a reliable baseline on `basic`, test the agent on `defend_the_center`.

**Objective:** Assess generalization to a more demanding environment.

Results from different scenarios should be reported separately.

### 5. Reproducibility and statistical evaluation

For meaningful comparisons:

- Evaluate each model over the same number of episodes.
- Use consistent success criteria.
- Report mean reward and success rate.
- Include variability across episodes and, where feasible, multiple training seeds.
- Record configuration parameters and checkpoint-selection rules.

---

## Project Structure

```text
doom_snn/
├── config.py          # Hyperparameters and configuration
├── env.py             # ViZDoom wrapper and observation preprocessing
├── model.py           # Spiking Q-network
├── replay.py          # Experience replay buffer
├── train.py           # DQN training and checkpointing
├── evaluate.py        # Policy evaluation and spike statistics
├── check_setup.py     # Environment and hardware checks
├── diagnose.py        # Failure-mode diagnostics
├── requirements.txt
├── requirements-gpu.txt
├── requirements-cpu.txt
├── logs/              # Training and evaluation logs
├── checkpoints/       # Saved model checkpoints
└── videos/            # Optional evaluation videos
```

The structure above describes the intended project layout; optional files and output directories may not exist until generated.

---

## Roadmap

- [x] Implement the initial SNN-DQN pipeline for ViZDoom.
- [x] Train and evaluate a baseline on `basic`.
- [x] Introduce the v2 training and checkpointing improvements.
- [x] Complete the v2 50,000-step training run.
- [ ] Diagnose instability using successful and failed episodes.
- [ ] Train and evaluate a comparable ANN/CNN baseline.
- [ ] Establish a reliable success-rate baseline.
- [ ] Ablate temporal simulation lengths \(T = 4, 8, 16\).
- [ ] Investigate firing-rate regularization.
- [ ] Evaluate on `defend_the_center`.
- [ ] Measure inference latency and characterize computational cost.
- [ ] Investigate INT8 weight quantization.
- [ ] Develop hardware-compatible network export.
- [ ] Validate the network on the target FPGA neuromorphic accelerator.

FPGA export and hardware deployment remain future work.

---

## Technologies

- **Python** — implementation and experiment management.
- **PyTorch** — neural-network training.
- **snnTorch** — spiking-neuron models and surrogate gradients.
- **ViZDoom** — game environment.
- **Gymnasium** — environment interface.
- **CUDA** — optional GPU acceleration.
- **Git** — version control.

---

## Research Focus

PeraMorphIQ explores the feasibility of combining spiking neural networks with deep reinforcement learning for autonomous visual control, with an emphasis on training stability, temporal computation, and eventual hardware deployment.

The immediate priority is to establish a reliable software baseline and determine whether the current performance limitations arise from the SNN implementation, the shared DQN pipeline, or the environment. Hardware efficiency and FPGA suitability will be evaluated in later stages rather than assumed from software-level spike counts alone.
