cd C:\Users\ayyas\doom_snn\doom_snn
@'
# Spiking Neural Network DOOM Agent (PeraMorphIQ)

A Python spiking neural network (SNN) pipeline that learns to play DOOM autonomously
(no human input) using ViZDoom. This is the software stage of a project whose end goal is
deployment on an FPGA neuromorphic accelerator, so the network uses only hardware-friendly
building blocks: convolution / linear layers and leaky integrate-and-fire (LIF) neurons
(no batch norm, no max-pooling).

**Status (20 Oct 2026):** the pipeline runs end to end on the `basic` scenario. The trained
agent has not yet learned to aim (see Key finding). Diagnosis is done; the fix is the next step.
Hardware export has not started.

## Method

- **Environment:** ViZDoom (`basic` scenario first), 160x120 grayscale frames resized to
  64x64, 4 stacked frames, frame-skip 4, three actions (move left, move right, attack).
- **Network (`model.py`, `SpikingDQN`):** Conv-LIF x3 -> FC-LIF -> non-spiking LIF readout.
  The readout membrane potential after T timesteps is used as the Q-values.
  Direct (constant-current) input encoding; surrogate gradients from snnTorch; fixed beta.
- **Learning (`train.py`):** DQN with experience replay, epsilon-greedy exploration,
  target network, Double DQN, Huber loss, gradient clipping and reward scaling (x0.01).
- **Evaluation (`evaluate.py`):** runs a trained policy with no human input and reports mean
  reward, success rate (reward > 0) and per-layer spike rates (activity is the energy proxy
  for neuromorphic hardware).
- **Diagnosis (`diagnose.py`):** action usage and Q-value spread for successful vs failed episodes.

## Results

v2 run: 50,000 steps on `basic`. Each checkpoint evaluated over 100 episodes (eps = 0.02).

| Checkpoint | Success rate | Mean reward |
|------------|--------------|-------------|
| v2_best (step 15k, picked by 30-episode eval) | 62% | -75.1 |
| v2_step30000 | 54% | -131.1 |
| v2_last (step 50k) | 70% | -55.9 |

Spike rates for v2_last (spikes/neuron/timestep): L1 0.21, L2 0.43, L3 0.51, L4 0.65.
This is dense activity; reducing it is a goal before hardware deployment.

Training history:

- Baseline (v1): learnable beta, 10-episode evals. Training curve oscillated between about
  -200 and +80 with no upward trend; 20-episode eval success was roughly 55-60%.
- v2: success 60% at 5k steps (while still mostly exploring), peak 77% at 15k (30-episode eval,
  an optimistic pick), then 47% at 50k on the same 30-episode eval. 100-episode numbers above
  are the reliable ones. A short 6-episode clip of v2_last scored 6/6 successes (used for the demo video).

### Key finding

`diagnose.py` on v2_last (30 episodes):

- The agent presses ATTACK about 99-100% of the time, in successful and failed episodes alike.
- Successful episodes last about 3 steps (the monster happened to spawn in the line of fire).
  Failed episodes last about 70 steps (shooting until the episode times out, reward about -410).
- So the 55-70% success rate is roughly the chance that the monster spawns in front of the
  agent, not learned aiming. Reward is bimodal: +95 or about -410.
- The Q-value spread is about 290 and identical at every step. With the reward scaling,
  Q-values should be within about +-4. The Q-values have exploded and the output no longer
  depends on the observation, which is why the greedy policy always picks the same action.

### Likely causes and next experiments

- Q-value divergence combined with exploration that decays to 5% by step 20k, so once the
  policy locks onto shooting it almost never tries moving.
- Next: (1) a plain ReLU CNN baseline with the same layer sizes to tell whether the problem is
  the spiking network or the DQN setup; (2) longer exploration, e.g.
  `--eps_end 0.2 --eps_decay_frac 0.8 --lr 2e-5`; (3) check Q-values stay bounded during training.

## Setup (Windows, Python 3.11)
