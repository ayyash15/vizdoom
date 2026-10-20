"""Spiking Q-network: Conv-LIF x3 -> FC-LIF -> non-spiking LIF readout (membrane = Q-values).

Direct (constant-current) input encoding: the first conv is computed once and injected
as current at every timestep. Only LIF neurons + conv/linear layers are used (no batch norm,
no max-pool), which keeps the model easy to map onto an FPGA neuromorphic accelerator.

v2: beta is fixed (learn_beta=False). A learnable beta can drift past 1.0 and destabilise
training, and a fixed beta is what the FPGA will implement anyway.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingDQN(nn.Module):
    def __init__(self, n_actions, in_ch=4, img=64, T=8, beta=0.9, hidden=256):
        super().__init__()
        self.T = T
        sg = surrogate.fast_sigmoid(slope=25)
        self.conv1 = nn.Conv2d(in_ch, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        with torch.no_grad():
            d = self.conv3(self.conv2(self.conv1(torch.zeros(1, in_ch, img, img))))
        flat = d.numel()
        self.fc1 = nn.Linear(flat, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

        kw = dict(beta=beta, spike_grad=sg, init_hidden=True, learn_beta=False)
        self.lif1, self.lif2 = snn.Leaky(**kw), snn.Leaky(**kw)
        self.lif3, self.lif4 = snn.Leaky(**kw), snn.Leaky(**kw)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg, init_hidden=True,
                                 output=True, reset_mechanism="none", learn_beta=False)
        self._lifs = [self.lif1, self.lif2, self.lif3, self.lif4, self.lif_out]

    def forward(self, x, return_rates=False, return_reg=False):
        """return_rates: also return per-layer spike rates as floats (for logging).
        return_reg: also return the mean spike rate as a differentiable tensor (for the
        firing-rate penalty in train.py)."""
        for m in self._lifs:
            m.reset_hidden()
        cur1 = self.conv1(x)
        rates = [0.0, 0.0, 0.0, 0.0]
        reg = 0.0
        mem = None
        for _ in range(self.T):
            s1 = self.lif1(cur1)
            s2 = self.lif2(self.conv2(s1))
            s3 = self.lif3(self.conv3(s2))
            s4 = self.lif4(self.fc1(s3.flatten(1)))
            _, mem = self.lif_out(self.fc2(s4))
            if return_rates:
                for i, s in enumerate((s1, s2, s3, s4)):
                    rates[i] = rates[i] + s.detach().mean().item() / self.T
            if return_reg:
                reg = reg + sum(s.mean() for s in (s1, s2, s3, s4)) / (4 * self.T)
        if return_rates:
            return mem, rates
        if return_reg:
            return mem, reg
        return mem


class CnnDQN(nn.Module):
    """Non-spiking baseline: same conv/linear sizes as SpikingDQN, ReLU, single forward pass.
    Used to check whether instability comes from the SNN or from the DQN setup."""

    def __init__(self, n_actions, in_ch=4, img=64, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        with torch.no_grad():
            d = self.conv3(self.conv2(self.conv1(torch.zeros(1, in_ch, img, img))))
        self.fc1 = nn.Linear(d.numel(), hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

    def forward(self, x, return_rates=False, return_reg=False):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        q = self.fc2(torch.relu(self.fc1(x.flatten(1))))
        if return_rates:
            return q, [float("nan")] * 4     # no spikes in a CNN
        if return_reg:
            return q, 0.0
        return q


def build_model(cfg, n_actions):
    """cfg can be a Config or the dict stored in a checkpoint (old checkpoints have no 'model' key)."""
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    if get("model", "snn") == "cnn":
        return CnnDQN(n_actions, get("stack"), get("img_size"), get("hidden"))
    return SpikingDQN(n_actions, get("stack"), get("img_size"), get("T"), get("beta"), get("hidden"))
