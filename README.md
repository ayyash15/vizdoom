# SNN plays DOOM (PeraMorphIQ) – software pipeline

Spiking DQN: Conv-LIF x3 -> FC-LIF -> LIF readout (membrane potential = Q-values),
trained end-to-end with surrogate gradients (snnTorch) on ViZDoom.

## Run (PowerShell, inside doom_snn_env)
    cd doom_snn
    python check_setup.py                          # verify GPU + ViZDoom + speed
    python train.py --scenario basic --total_steps 50000        # quick first run
    python train.py --scenario defend_the_center --total_steps 300000
    python evaluate.py --ckpt checkpoints\basic_best.pt --episodes 5 --render --video

Any Config field can be overridden: `--T 4 --batch_size 16 --buffer_size 15000`.

## Resources (your machine: RTX 3050 Laptop, ~14 GB free disk)
- VRAM: well under 1 GB.  RAM: ~0.6 GB replay buffer.  Disk: a few MB (checkpoints ~2 MB each).
- Out of memory? lower `--buffer_size` or `--batch_size`. Too slow? lower `--T` (e.g. 4).

## Hardware-friendly design choices
- Only LIF neurons, conv, linear layers (no batch-norm / max-pool / attention).
- Direct input encoding; readout is a non-spiking LIF integrator.
- `evaluate.py` reports per-layer spike rates (activity = energy proxy for the FPGA).

## Ideas to improve the score
Curriculum over scenarios, Double/Dueling DQN, reward shaping, rate/latency encoding,
weight quantization (int8) + beta/threshold fixed for FPGA export, PPO/recurrent variants.
