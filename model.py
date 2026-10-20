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

    def forward(self, x, return_rates=False):
        for m in self._lifs:
            m.reset_hidden()
        cur1 = self.conv1(x)
        rates = [0.0, 0.0, 0.0, 0.0]
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
        return (mem, rates) if return_rates else mem
