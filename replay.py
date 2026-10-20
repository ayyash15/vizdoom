"""Memory-light replay buffer: stores the stacked state + only the newest next-frame (uint8)."""
import numpy as np
import torch


class Replay:
    def __init__(self, capacity, stack, size):
        self.cap, self.i, self.n = capacity, 0, 0
        self.s = np.zeros((capacity, stack, size, size), np.uint8)
        self.nf = np.zeros((capacity, 1, size, size), np.uint8)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, next_state, done):
        i = self.i
        self.s[i], self.nf[i, 0] = s, next_state[-1]
        self.a[i], self.r[i], self.d[i] = a, r, float(done)
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, device):
        idx = np.random.randint(0, self.n, batch)
        s = self.s[idx]
        ns = np.concatenate([s[:, 1:], self.nf[idx]], axis=1)
        f = lambda x: torch.from_numpy(x).to(device)
        return (f(s).float() / 255, f(self.a[idx]), f(self.r[idx]),
                f(ns).float() / 255, f(self.d[idx]))
