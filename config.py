"""Central configuration. Phase 1 = CNN-DQN baseline (target >90% success on basic). Phase 2 = SNN."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- environment ---
    scenario: str = "basic"        # basic | defend_the_center | health_gathering | defend_the_line | deadly_corridor
    frame_skip: int = 2
    img_size: int = 64             # frames resized to img_size x img_size
    stack: int = 4                 # frames stacked as input channels
    reward_scale: float = 0.01     # raw ViZDoom rewards are large (e.g. +101), scale for stable learning

    # --- model ---
    model: str = "cnn"             # cnn | snn  (PHASE 1: cnn. Switch to snn only after cnn passes 90%)
    rate_reg: float = 0.0          # firing-rate penalty weight (SNN only). 0 = off. Try 0.01 - 0.1

    # --- SNN ---
    T: int = 8                     # simulation timesteps per decision
    beta: float = 0.9              # LIF membrane decay (now FIXED, not learned -> stable + FPGA friendly)
    hidden: int = 256

    # --- DQN ---
    total_steps: int = 250_000     # agent decisions (each = frame_skip game tics)
    batch_size: int = 32
    lr: float = 5e-5               # was 1e-4
    gamma: float = 0.99
    buffer_size: int = 30_000      # ~20 KB/transition -> ~600 MB RAM
    warmup: int = 1_000
    train_every: int = 2
    target_update: int = 1_000
    double_dqn: int = 1            # 1 = Double DQN (less Q overestimation), 0 = vanilla
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frac: float = 0.4    # fraction of total_steps for linear decay
    grad_clip: float = 10.0

    # --- evaluation / bookkeeping ---
    eval_every: int = 5_000
    eval_episodes: int = 30        # was 10 (too noisy to pick a "best" checkpoint)
    eval_eps: float = 0.00        # tiny exploration at eval so the agent can't get stuck in a fixed loop
    run_name: str = "cnn_tuned"             # tag for logs/checkpoints; empty = scenario name
    seed: int = 42
