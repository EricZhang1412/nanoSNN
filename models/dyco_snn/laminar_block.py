from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .glif3_neuron import GLIF3Population, GLIFState
from .recurrent import DaleLinear, make_sign_vec


@dataclass
class BlockState:
    s_l4: GLIFState
    s_23e: GLIFState
    s_23i: GLIFState
    s_l5: GLIFState
    z_23e_prev: torch.Tensor  # for recurrence delay


class LaminarBlock(nn.Module):
    """
    Compact laminar E/I micro-circuit: L4 (input layer) -> L2/3 (E+I,
    locally recurrent) -> L5/6 (output layer). All within-block projections
    obey Dale's law on E/I separation.

    Channel D = per-population GLIF3 intrinsic dynamics.
    Channel C = L2/3 -> L2/3 recurrent projection (Dale).

    Allocation alpha (per-block scalar via parent module) modulates:
      - recurrent current path gain (scaled by alpha)
      - intrinsic SFA strength via asc_amps_scale (scaled by 1 - alpha)
    """

    def __init__(
        self,
        in_features: int,
        n_l4: int = 48,
        n_23e: int = 64,
        n_23i: int = 16,
        n_l5: int = 48,
        learnable_tau: bool = True,
        learnable_asc: bool = True,
        heterogeneous_init: bool = True,
        recurrent_enabled: bool = True,
        rec_scale_init: float = 0.5,
        proj_scale_init: float = 30.0,
        v_th: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_l4 = n_l4
        self.n_23e = n_23e
        self.n_23i = n_23i
        self.n_l5 = n_l5
        self.n_23 = n_23e + n_23i
        self.recurrent_enabled = recurrent_enabled

        # Within-block feedforward projections (no Dale constraint on stem
        # inputs since input feature signs are unconstrained; L4 itself is
        # all-excitatory, so its outward projections are positive).
        # Default init magnitude is scaled up so neurons actually reach v_th
        # under the default GLIF3 time constants (tau_m~5-15 hetero, tau_s~2-10).
        self.proj_in_to_l4 = nn.Linear(in_features, n_l4, bias=False)
        self.proj_l4_to_23 = nn.Linear(n_l4, self.n_23, bias=False)
        self.proj_23_to_l5 = nn.Linear(self.n_23, n_l5, bias=False)
        with torch.no_grad():
            for p in (self.proj_in_to_l4.weight,
                      self.proj_l4_to_23.weight,
                      self.proj_23_to_l5.weight):
                p.mul_(proj_scale_init)

        # L2/3 recurrence (Dale): inputs are L2/3 E+I spikes; output is
        # current into L2/3 (E + I targets all share this output).
        sign_vec = make_sign_vec(n_23e, n_23i)
        self.rec_23 = DaleLinear(
            in_features=self.n_23, out_features=self.n_23, sign_vec=sign_vec,
            scale_init=rec_scale_init,
        )
        if not recurrent_enabled:
            for p in self.rec_23.parameters():
                p.requires_grad_(False)
                with torch.no_grad():
                    p.zero_()

        # Populations.
        pop_kwargs = dict(
            learnable_tau=learnable_tau,
            learnable_asc=learnable_asc,
            heterogeneous_init=heterogeneous_init,
            v_th=v_th,
        )
        self.pop_l4 = GLIF3Population(n_l4, **pop_kwargs)
        self.pop_23e = GLIF3Population(n_23e, **pop_kwargs)
        self.pop_23i = GLIF3Population(n_23i, **pop_kwargs)
        self.pop_l5 = GLIF3Population(n_l5, **pop_kwargs)

    def set_alpha(self, alpha: torch.Tensor):
        """alpha in [0,1]. SFA scale = (1-alpha); recurrent scale applied
        explicitly in forward()."""
        self._alpha = alpha
        sfa_scale = (1.0 - alpha).detach() if not alpha.requires_grad else (1.0 - alpha)
        for pop in (self.pop_l4, self.pop_23e, self.pop_23i, self.pop_l5):
            pop.set_asc_scale(sfa_scale)

    def zero_state(self, batch_size: int, device, dtype=torch.float32) -> BlockState:
        return BlockState(
            s_l4=self.pop_l4.zero_state(batch_size, device, dtype),
            s_23e=self.pop_23e.zero_state(batch_size, device, dtype),
            s_23i=self.pop_23i.zero_state(batch_size, device, dtype),
            s_l5=self.pop_l5.zero_state(batch_size, device, dtype),
            z_23e_prev=torch.zeros(batch_size, self.n_23e, device=device, dtype=dtype),
        )

    def step(self, x_t: torch.Tensor, state: BlockState):
        """One timestep. x_t: [B, in_features] (current into L4)."""
        alpha = self._alpha

        # L4: feedforward from upstream.
        l4_drive = self.proj_in_to_l4(x_t)
        z_l4, s_l4 = self.pop_l4(l4_drive, state.s_l4)

        # L2/3: feedforward from L4 + (alpha * recurrent from previous L2/3).
        l23_ff = self.proj_l4_to_23(z_l4)
        # Recurrence uses previous-step L2/3 (E+I) spikes.
        z_23_prev_full = torch.cat([state.z_23e_prev,
                                    torch.zeros(state.z_23e_prev.shape[0], self.n_23i,
                                                device=x_t.device, dtype=x_t.dtype)], dim=-1)
        rec_current = self.rec_23(z_23_prev_full) if self.recurrent_enabled else 0.0
        l23_drive = l23_ff + alpha * rec_current
        # Split into E and I targets.
        l23_drive_e, l23_drive_i = l23_drive[:, :self.n_23e], l23_drive[:, self.n_23e:]
        z_23e, s_23e = self.pop_23e(l23_drive_e, state.s_23e)
        z_23i, s_23i = self.pop_23i(l23_drive_i, state.s_23i)
        z_23 = torch.cat([z_23e, z_23i], dim=-1)

        # L5/6: receives L2/3.
        l5_drive = self.proj_23_to_l5(z_23)
        z_l5, s_l5 = self.pop_l5(l5_drive, state.s_l5)

        new_state = BlockState(
            s_l4=s_l4, s_23e=s_23e, s_23i=s_23i, s_l5=s_l5,
            z_23e_prev=z_23e,
        )
        return z_l5, new_state

    def forward(self, x_seq: torch.Tensor):
        """x_seq: [T, B, in_features]. Returns [T, B, n_l5]."""
        T, B = x_seq.shape[0], x_seq.shape[1]
        state = self.zero_state(B, x_seq.device, x_seq.dtype)
        outs = []
        for t in range(T):
            z, state = self.step(x_seq[t], state)
            outs.append(z)
        return torch.stack(outs, dim=0)
