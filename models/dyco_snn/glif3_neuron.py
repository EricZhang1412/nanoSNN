from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeGauss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled, sigma, amplitude):
        ctx.save_for_backward(v_scaled, sigma, amplitude)
        return (v_scaled > 0).to(v_scaled.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        v_scaled, sigma, amplitude = ctx.saved_tensors
        grad = torch.exp(-(v_scaled ** 2) / (sigma ** 2)) * amplitude
        return grad_output * grad, None, None


def spike_gauss(v_scaled, sigma, amplitude):
    return SpikeGauss.apply(v_scaled, sigma, amplitude)


@dataclass
class GLIFState:
    v: torch.Tensor          # [B, N]
    asc1: torch.Tensor       # [B, N]
    asc2: torch.Tensor       # [B, N]
    psc: torch.Tensor        # [B, N]  (single-receptor PSC)
    refrac: torch.Tensor     # [B, N]


class GLIF3Population(nn.Module):
    """
    Per-population GLIF3-style adaptive LIF.

    Per-population learnable params (when enabled):
      log_tau_m, log_tau_s, asc_amps (size 2), log_param_k (size 2).

    Membrane (normalized, all units dt=1):
      v_{t+1} = decay_m * v_t + cf_m * I_t + reset_t
      I_t    = psc_t + asc1_t + asc2_t + (1-alpha) * d_drive_t
      psc_{t+1} = decay_s * psc_t + (1 - decay_s) * input_current_t
      asc_{t+1} = decay_k * asc_t + amp * z_t

    'alpha' scales how much *recurrent* current is allowed in vs base
    drive (handled by caller). Intrinsic SFA strength is scaled by
    (1 - alpha) outside via asc_amps_scale.

    All state lives in `GLIFState`. This module is stateless across calls.
    """

    def __init__(
        self,
        n: int,
        learnable_tau: bool = True,
        learnable_asc: bool = True,
        heterogeneous_init: bool = True,
        tau_m_init: float = 20.0,
        tau_s_init: float = 5.0,
        tau_m_range: tuple = (10.0, 40.0),
        tau_s_range: tuple = (2.0, 10.0),
        asc_amps_init: tuple = (-0.05, -0.02),
        param_k_init: tuple = (0.1, 0.01),
        v_th: float = 1.0,
        v_reset: float = 0.0,
        t_ref: float = 2.0,
        gauss_std: float = 0.5,
        dampening_factor: float = 0.3,
        asc_amps_scale: float = 1.0,
    ):
        super().__init__()
        self.n = int(n)
        self.v_th = float(v_th)
        self.v_reset = float(v_reset)
        self.t_ref = float(t_ref)

        # Per-population log time constants.
        if heterogeneous_init:
            log_m_lo, log_m_hi = math.log(tau_m_range[0]), math.log(tau_m_range[1])
            log_s_lo, log_s_hi = math.log(tau_s_range[0]), math.log(tau_s_range[1])
            log_tau_m = torch.empty(n).uniform_(log_m_lo, log_m_hi)
            log_tau_s = torch.empty(n).uniform_(log_s_lo, log_s_hi)
        else:
            log_tau_m = torch.full((n,), math.log(tau_m_init))
            log_tau_s = torch.full((n,), math.log(tau_s_init))

        self.log_tau_m = nn.Parameter(log_tau_m, requires_grad=learnable_tau)
        self.log_tau_s = nn.Parameter(log_tau_s, requires_grad=learnable_tau)

        asc_amps = torch.tensor([asc_amps_init[0], asc_amps_init[1]], dtype=torch.float32)
        asc_amps = asc_amps.unsqueeze(0).repeat(n, 1)  # [N, 2]
        log_param_k = torch.tensor(
            [math.log(param_k_init[0]), math.log(param_k_init[1])], dtype=torch.float32
        )
        log_param_k = log_param_k.unsqueeze(0).repeat(n, 1)  # [N, 2]

        self.asc_amps = nn.Parameter(asc_amps, requires_grad=learnable_asc)
        self.log_param_k = nn.Parameter(log_param_k, requires_grad=learnable_asc)
        # External scale to disable / dampen SFA (used by alpha gate at block level).
        self.register_buffer("asc_amps_scale", torch.tensor(float(asc_amps_scale)))

        self.register_buffer("gauss_std", torch.tensor(float(gauss_std)))
        self.register_buffer("dampening_factor", torch.tensor(float(dampening_factor)))

    def set_asc_scale(self, scale):
        """Scale SFA strength at runtime (tensor or float). Used by alpha gate."""
        if not torch.is_tensor(scale):
            scale = torch.tensor(float(scale), device=self.asc_amps_scale.device)
        self.asc_amps_scale = scale.to(self.asc_amps_scale.device).to(self.asc_amps_scale.dtype)

    def zero_state(self, batch_size: int, device, dtype=torch.float32) -> GLIFState:
        z = lambda: torch.zeros(batch_size, self.n, device=device, dtype=dtype)
        return GLIFState(v=z(), asc1=z(), asc2=z(), psc=z(), refrac=z())

    def forward(self, input_current: torch.Tensor, state: GLIFState):
        """
        input_current: [B, N]    — synaptic drive into this population at time t.
        Returns (spikes z [B, N], new_state).
        """
        decay_m = torch.exp(-1.0 / torch.exp(self.log_tau_m).clamp_min(1e-3))  # [N]
        decay_s = torch.exp(-1.0 / torch.exp(self.log_tau_s).clamp_min(1e-3))  # [N]
        decay_k = torch.exp(-torch.exp(self.log_param_k).clamp_min(1e-3))      # [N, 2]
        cf_m = (1.0 - decay_m)                                                  # current factor

        # PSC integrates input current with synapse time constant.
        new_psc = decay_s[None, :] * state.psc + (1.0 - decay_s[None, :]) * input_current

        # After-spike currents (scaled by external alpha gate).
        amps = self.asc_amps * self.asc_amps_scale  # [N, 2]
        new_asc1 = decay_k[None, :, 0] * state.asc1
        new_asc2 = decay_k[None, :, 1] * state.asc2

        # Refractory countdown.
        new_refrac = F.relu(state.refrac - 1.0)
        not_refrac = (new_refrac <= 0.0).float()

        # Membrane update.
        total_current = new_psc + new_asc1 + new_asc2
        new_v = decay_m[None, :] * state.v + cf_m[None, :] * total_current

        v_scaled = (new_v - self.v_th) / max(self.v_th - self.v_reset, 1e-3)
        z = spike_gauss(v_scaled, self.gauss_std, self.dampening_factor) * not_refrac

        # Apply spike contributions to next-state ASC and reset membrane.
        new_asc1 = new_asc1 + z * amps[:, 0][None, :]
        new_asc2 = new_asc2 + z * amps[:, 1][None, :]
        new_v = torch.where(z > 0, torch.full_like(new_v, self.v_reset), new_v)
        new_refrac = torch.where(z > 0, torch.full_like(new_refrac, self.t_ref), new_refrac)

        return z, GLIFState(v=new_v, asc1=new_asc1, asc2=new_asc2, psc=new_psc, refrac=new_refrac)
