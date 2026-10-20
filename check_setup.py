"""Sanity check: GPU, imports, ViZDoom random episode, SNN forward/backward speed."""
import time, random
import torch
from env import DoomEnv
from model import SpikingDQN

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = DoomEnv("basic")
s, done, steps = env.reset(), False, 0
while not done:
    s, r, done = env.step(random.randrange(env.n_actions)); steps += 1
print(f"ViZDoom OK: random episode finished in {steps} decisions, obs {s.shape}")

net = SpikingDQN(env.n_actions).to(dev)
x = torch.rand(32, 4, 64, 64, device=dev)
for _ in range(2): net(x).sum().backward()
if dev.type == "cuda": torch.cuda.synchronize()
t = time.time()
for _ in range(10): net(x).sum().backward()
if dev.type == "cuda": torch.cuda.synchronize()
print(f"SNN fwd+bwd (batch 32, T=8): {(time.time()-t)/10*1000:.0f} ms/iter")
if dev.type == "cuda": print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
print("All good.")
