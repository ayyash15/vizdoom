"""Central configuration. Defaults are tuned for an RTX 3050 Laptop GPU (4 GB VRAM)."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- environment ---
    scenario: str = "basic"        # basic | defend_the_center | health_gathering | defend_the_line | deadly_corridor
    frame_skip: int = 4
    img_size: int = 64             # frames resized to img_size x img_size
    stack: int = 4                 # frames stacked as input channels
    reward_scale: float = 0.01     # raw ViZDoom rewards are large (e.g. +101), scale for stable learning

    # --- SNN ---
    T: int = 8                     # simulation timesteps per decision
    beta: float = 0.9              # LIF membrane decay
    hidden: int = 256

    # --- DQN ---
    total_steps: int = 150_000     # agent decisions (each = frame_skip game tics)
    batch_size: int = 32
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 30_000      # ~20 KB/transition -> ~600 MB RAM
    warmup: int = 1_000
    train_every: int = 2
    target_update: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4    # fraction of total_steps for linear decay
    grad_clip: float = 10.0

    # --- misc ---
    eval_every: int = 10_000
    eval_episodes: int = 10
    seed: int = 42
