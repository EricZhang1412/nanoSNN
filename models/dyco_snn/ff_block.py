from __future__ import annotations

import math

import torch
import torch.nn as nn

from .glif3_neuron import GLIF3Population


class FeedforwardBlock(nn.Module):
    """
    Param-matched feedforward baseline: stack of Linear+LIF with no
    recurrence and no SFA. Hidden width auto-set to match a laminar
    block's parameter budget when target_params is given.

    Block params (laminar reference: in*n_l4 + n_l4*(n_23e+n_23i)
      + (n_23e+n_23i)*n_l5 + (n_23e+n_23i)^2 + 4*(per_pop params))
    FF block: in*h1 + h1*h2 + h2*out where h2==out==n_l5.
    """

    def __init__(self, in_features: int, n_l5: int = 48,
                 hidden_mult: float = 2.0, depth: int = 2,
                 proj_scale_init: float = 30.0):
        super().__init__()
        h = max(int(round(in_features * hidden_mult)), n_l5)
        dims = [in_features] + [h] * (depth - 1) + [n_l5]
        layers = []
        pops = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1], bias=False)
            with torch.no_grad():
                lin.weight.mul_(proj_scale_init)
            layers.append(lin)
            pops.append(GLIF3Population(
                dims[i + 1],
                learnable_tau=False, learnable_asc=False,
                heterogeneous_init=False,
                # Zero out SFA for plain LIF behavior.
                asc_amps_init=(0.0, 0.0),
            ))
        self.linears = nn.ModuleList(layers)
        self.pops = nn.ModuleList(pops)
        self.n_l5 = n_l5

    def set_alpha(self, alpha):
        # FF block has no alpha semantics; accept and ignore.
        self._alpha = alpha

    def forward(self, x_seq: torch.Tensor):
        T, B = x_seq.shape[0], x_seq.shape[1]
        states = [p.zero_state(B, x_seq.device, x_seq.dtype) for p in self.pops]
        outs = []
        for t in range(T):
            h = x_seq[t]
            for lin, pop, i in zip(self.linears, self.pops, range(len(self.pops))):
                drive = lin(h)
                z, states[i] = pop(drive, states[i])
                h = z
            outs.append(h)
        return torch.stack(outs, dim=0)
