"""Central configuration. Phase 1 = CNN-DQN baseline (basic: ceiling is ~84 mean reward). Phase 2 = SNN."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- environment ---
    scenario: str = "basic"        # basic | defend_the_center | health_gathering | defend_the_line | deadly_corridor
    frame_skip: int = 2            # was 4. Finer control; ceiling gain ~+0.7 on basic
    img_size: int = 64             # frames resized to img_size x img_size
    stack: int = 4                 # frames stacked as input channels
    reward_scale: float = 0.01     # raw ViZDoom rewards are large (e.g. +101), scale for stable learning
    combo_actions: int = 0         # 1 = add "move+attack" actions (assumes ATTACK is the LAST button, true for basic)

    # --- aiming action space (experimental; "strafe" = original baseline, unchanged) ---
    aim_mode: str = "strafe"       # strafe | turn | delta
    aim_deltas: str = "1,3,8,20"   # delta mode: signed magnitudes (+/-) of TURN_LEFT_RIGHT_DELTA, comma separated
    aim_unit: float = 1.0          # delta mode: multiplier on aim_deltas (UNVERIFIED units; calibrate with oracle_aim.py)
    # combo_actions=1 in delta mode adds one "turn d + ATTACK" action per signed delta

    # --- model ---
    model: str = "cnn"             # cnn | snn  (PHASE 1: cnn. Switch to snn only after cnn passes)
    rate_reg: float = 0.0          # firing-rate penalty weight (SNN only). 0 = off. Try 0.01 - 0.1

    # --- SNN ---
    T: int = 8                     # simulation timesteps per decision
    beta: float = 0.9              # LIF membrane decay (FIXED, not learned -> stable + FPGA friendly)
    hidden: int = 256

    # --- DQN ---
    total_steps: int = 300_000     # agent decisions (each = frame_skip game tics)
    batch_size: int = 32
    lr: float = 5e-5
    lr_decay: int = 1              # 1 = cosine decay of lr over training
    lr_final_frac: float = 0.1     # final lr = lr * lr_final_frac
    gamma: float = 0.99
    buffer_size: int = 30_000      # ~20 KB/transition -> ~600 MB RAM
    warmup: int = 1_000
    train_every: int = 2
    target_update: int = 1_000
    double_dqn: int = 1            # 1 = Double DQN, 0 = vanilla
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frac: float = 0.4    # fraction of total_steps for linear decay
    grad_clip: float = 10.0

    # --- evaluation / bookkeeping ---
    eval_every: int = 10_000
    eval_episodes: int = 50        # per periodic eval
    eval_eps: float = 0.00         # exploration at eval (0 = pure greedy)
    final_top_k: int = 3           # re-score this many best checkpoints at the end
    final_eval_episodes: int = 200 # episodes for that re-scoring (same spawns for every candidate)
    run_name: str = "cnn_fs2"      # tag for logs/checkpoints; empty = scenario name
    seed: int = 42