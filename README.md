# Spiking Neural Network DOOM Agent (PeraMorphIQ)

A Python spiking neural network (SNN) pipeline that learns to play DOOM autonomously
(no human input) using ViZDoom. This is the software stage of a project whose end goal is
deployment on an FPGA neuromorphic accelerator, so the network uses only hardware-friendly
building blocks: convolution / linear layers and leaky integrate-and-fire (LIF) neurons
(no batch norm, no max-pooling).

**Status (20 Oct 2026):** baseline agent trained on the `basic` scenario; v2 training
pipeline committed and validation run in progress. Hardware export is not started yet.

## Method

- **Environment:** ViZDoom (`basic` scenario first), 160x120 grayscale frames resized to
  64x64, 4 stacked frames, frame-skip 4.
- **Network (`model.py`, `SpikingDQN`):** Conv-LIF x3 -> FC-LIF -> non-spiking LIF readout.
  The readout's membrane potential after T timesteps is used as the Q-values.
  Input uses direct (constant-current) encoding; surrogate gradients from snnTorch.
- **Learning (`train.py`):** DQN with experience replay, epsilon-greedy exploration and a
  target network, trained with Huber loss, gradient clipping and reward scaling.
- **Evaluation (`evaluate.py`):** runs the trained policy with no human input and reports
  mean reward, success rate and the per-layer spike rate (activity is the energy proxy
  for neuromorphic hardware).

## Results so far

| Run | Scenario | Steps | Result |
|-----|----------|-------|--------|
| Baseline (v1) | basic | 50k (resumed) | Unstable. Eval over 20 episodes: mean reward -76.5 (best checkpoint) and -67.9 (last); episodes are bimodal (+95 success vs -410 timeout), roughly 55-60% success. Training curve oscillates between -200 and +80 with no stable upward trend. |
| v2 | basic | 50k | In progress |

Baseline spike rates (spikes/neuron/timestep): L1 0.25, L2 0.40, L3 0.58, L4 0.61, which is
denser than ideal for neuromorphic hardware.

### Diagnosis of the baseline

- Learnable LIF decay (`beta`) could drift during training.
- Evaluation used 10 episodes, too noisy to pick a "best" checkpoint when episode rewards
  swing between +95 and -410.
- Greedy evaluation with no randomness probably lets the agent get stuck repeating one
  action until the episode times out (hypothesis, being checked with v2).

### v2 changes

- Fixed (non-learned) `beta`, which is also what the FPGA will implement.
- Double DQN and a lower learning rate (5e-5).
- Evaluation every 5k steps over 30 episodes, with a tiny eval epsilon (0.02).
- Best checkpoint chosen by success rate (reward > 0), and a checkpoint saved at every eval.
- `--run_name` tag so runs do not overwrite each other's logs and checkpoints.

## Setup (Windows, Python 3.11)

```
python -m venv doom_snn_env
doom_snn_env\Scripts\Activate.ps1
pip install -r requirements.txt
python check_setup.py
```

Tested with torch 2.6.0 (CUDA 12.4 build), snntorch 1.0.0, vizdoom 1.3.2, gymnasium 1.4.0 on an
RTX 3050 Laptop GPU (4 GB). Use the High performance power plan on laptops: the GPU crashed
once under the Balanced plan during sustained training.

## Usage

Train:

```
python train.py --scenario basic --total_steps 50000 --run_name v2
```

Evaluate a checkpoint (success rate and spike rates are printed):

```
python evaluate.py --ckpt checkpoints\v2_best.pt --episodes 30
python evaluate.py --ckpt checkpoints\v2_best.pt --episodes 3 --render --video
```

Outputs: `logs/<run>_episodes.csv`, `logs/<run>_evals.csv`, `logs/<run>_curve.png`,
`checkpoints/<run>_best.pt` (not tracked in git), `videos/` (not tracked in git).

## Files

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters (all overridable from the command line) |
| `env.py` | ViZDoom wrapper: grayscale, resize, frame-skip, frame-stack |
| `model.py` | Spiking Q-network |
| `replay.py` | Memory-light replay buffer |
| `train.py` | DQN training loop, evaluation, checkpointing |
| `evaluate.py` | Run a trained policy, report reward, success rate, spike rates |
| `check_setup.py` | Verify GPU, ViZDoom and speed before a run |

## Next steps

1. Validate v2 on `basic` (target: stable success rate above 80%).
2. Ablate the number of timesteps T (4 / 8 / 16) and report reward versus latency.
3. Report and reduce spike activity (firing-rate regularisation).
4. Train on a harder scenario (`defend_the_center`).
5. Later: int8 weight quantisation and export for the FPGA accelerator.
