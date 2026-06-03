from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DaleLinear(nn.Module):
    """
    Linear projection that respects Dale's law: each presynaptic unit is
    either excitatory (+) or inhibitory (-), and per-synapse weight sign is
    frozen to the population sign.

    sign_vec: [in_features] in {+1, -1}.
    Effective weight = sign_vec[None, :] * relu(raw_weight).
    """

    def __init__(self, in_features: int, out_features: int, sign_vec: torch.Tensor,
                 bias: bool = False, scale_init: float = 0.1):
        super().__init__()
        if sign_vec.numel() != in_features:
            raise ValueError(f"sign_vec size {sign_vec.numel()} != in_features {in_features}")
        w = torch.empty(out_features, in_features).uniform_(0.0, scale_init)
        self.weight = nn.Parameter(w)
        self.register_buffer("sign_vec", sign_vec.float().view(1, -1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        return F.relu(self.weight) * self.sign_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


def make_sign_vec(n_e: int, n_i: int) -> torch.Tensor:
    return torch.cat([torch.ones(n_e), -torch.ones(n_i)], dim=0)
