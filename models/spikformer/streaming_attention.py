from __future__ import annotations

import types
from typing import Any

import torch

from .gate_attention import _bn1d

_GATED_ATTN_TYPES = {"c0_sdla", "c1_lowrank", "c2_oneminusk", "c3_mga"}


def _project_out(module: Any, out: torch.Tensor, T: int, B: int) -> torch.Tensor:
    # out: [T, B, H, N, D] -> [T, B, N, C]
    _, _, H, N, D = out.shape
    out = out.permute(0, 1, 3, 2, 4).contiguous().reshape(T, B, N, H * D)
    out = module.attn_lif(out)
    out = module.proj_linear(out.reshape(T * B, N, H * D))
    out = _bn1d(out.reshape(T, B, N, H * D), module.proj_bn)
    out = module.proj_lif(out)
    return out




def _compute_c3_gates(module: Any, k: torch.Tensor, k_membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T, B, H, N, D = k.shape
    tau_gamma, V_gamma, tau_beta, V_beta = module._gate_params()
    alpha_gamma = 1.0 - 1.0 / tau_gamma
    alpha_beta = 1.0 - 1.0 / tau_beta
    u_gamma = k.new_zeros(B, H, D)
    u_beta = k.new_zeros(B, H)
    gamma_fire_sum = k.new_zeros(())
    beta_fire_sum = k.new_zeros(())
    gamma_seq = []
    beta_seq = []

    for t in range(T):
        if module.gamma_input == "membrane":
            u_k_pool_t = k_membrane[t].mean(dim=-2)
        elif module.gamma_input == "spike":
            u_k_pool_t = k[t].float().mean(dim=-2)
        else:
            u_k_pool_t = k[t].float().mean(dim=(-2, -1)).unsqueeze(-1).expand(-1, -1, D)
        if module.gate_input_norm is not None and module.gamma_input != "rate":
            u_k_pool_t = module.gate_input_norm(u_k_pool_t)
        r_pool_t = k[t].float().mean(dim=(-2, -1))

        if module.use_gamma_gate:
            u_gamma = alpha_gamma * u_gamma + u_k_pool_t
            s_gamma = module._gate_surrogate(u_gamma - V_gamma)
            s_gamma_reset = s_gamma.detach() if module._k_detach_reset else s_gamma
            u_gamma = u_gamma - s_gamma_reset * V_gamma
        else:
            s_gamma = k.new_zeros(B, H, D)

        if module.use_beta_gate:
            u_beta = alpha_beta * u_beta + r_pool_t
            s_beta = module._gate_surrogate(u_beta - V_beta)
            s_beta_reset = s_beta.detach() if module._k_detach_reset else s_beta
            u_beta = u_beta - s_beta_reset * V_beta
        else:
            s_beta = k.new_ones(B, H)

        gamma_fire_sum = gamma_fire_sum + s_gamma.detach().float().mean()
        beta_fire_sum = beta_fire_sum + s_beta.detach().float().mean()
        gamma_seq.append(s_gamma)
        beta_seq.append(s_beta)

    with torch.no_grad():
        module.last_gamma_rate.copy_(gamma_fire_sum / float(T))
        module.last_beta_rate.copy_(beta_fire_sum / float(T))
    return torch.stack(gamma_seq, dim=0), torch.stack(beta_seq, dim=0)


def _triton_forward(module: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_membrane: torch.Tensor | None, attn_type: str) -> torch.Tensor:
    from .triton_attention import streaming_linear_attention_fwd

    if attn_type == "c0_sdla":
        return streaming_linear_attention_fwd(q, k, v, mode="c0", scale=float(module.scale))
    if attn_type == "c1_lowrank":
        gate = torch.sigmoid(module.gate_down(module.gate_up(k.mean(dim=-2))))
        return streaming_linear_attention_fwd(q, k, v, mode="diag", gate_d=gate, scale=float(module.scale))
    if attn_type == "c2_oneminusk":
        gate = 1.0 - k.float().mean(dim=-2)
        return streaming_linear_attention_fwd(q, k, v, mode="diag", gate_d=gate, scale=float(module.scale))
    if attn_type == "c3_mga":
        if k_membrane is None:
            raise RuntimeError("C3 Triton streaming attention requires k_membrane")
        s_gamma, s_beta = _compute_c3_gates(module, k, k_membrane)
        write_scale = torch.exp(module.log_write_scale)
        return streaming_linear_attention_fwd(
            q,
            k,
            v,
            mode="mga",
            gate_d=s_gamma,
            gate_scalar=s_beta,
            write_scale=write_scale,
            scale=float(module.scale),
            shift_scale=float(module.shift_scale),
        )
    raise RuntimeError(f"Unsupported attention type for Triton streaming: {attn_type}")


def _forward_c0(module: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    T = q.shape[0]
    outputs = []
    for t in range(T):
        kv_t = k[t].transpose(-2, -1) @ v[t]
        outputs.append((q[t] @ kv_t) * module.scale)
    return torch.stack(outputs, dim=0)


def _forward_c1(module: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    T, B, H, _, D = k.shape
    S_prev = k.new_zeros(B, H, D, D)
    outputs = []
    for t in range(T):
        k_bar_t = k[t].mean(dim=-2)
        gate_t = torch.sigmoid(module.gate_down(module.gate_up(k_bar_t)))
        kv_t = k[t].transpose(-2, -1) @ v[t]
        S_prev = gate_t.unsqueeze(-1) * S_prev + kv_t
        outputs.append((q[t] @ S_prev) * module.scale)
    return torch.stack(outputs, dim=0)


def _forward_c2(module: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    T, B, H, _, D = k.shape
    S_prev = k.new_zeros(B, H, D, D)
    outputs = []
    for t in range(T):
        gate_t = 1.0 - k[t].float().mean(dim=-2)
        kv_t = k[t].transpose(-2, -1) @ v[t]
        S_prev = gate_t.unsqueeze(-1) * S_prev + kv_t
        outputs.append((q[t] @ S_prev) * module.scale)
    return torch.stack(outputs, dim=0)


def _forward_c3(module: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_membrane: torch.Tensor) -> torch.Tensor:
    T, B, H, N, D = k.shape
    tau_gamma, V_gamma, tau_beta, V_beta = module._gate_params()
    alpha_gamma = 1.0 - 1.0 / tau_gamma
    alpha_beta = 1.0 - 1.0 / tau_beta
    write_scale = torch.exp(module.log_write_scale).view(1, H, 1, 1)

    S_prev = k.new_zeros(B, H, D, D)
    u_gamma = k.new_zeros(B, H, D)
    u_beta = k.new_zeros(B, H)
    gamma_fire_sum = k.new_zeros(())
    beta_fire_sum = k.new_zeros(())
    outputs = []

    for t in range(T):
        if module.gamma_input == "membrane":
            u_k_pool_t = k_membrane[t].mean(dim=-2)
        elif module.gamma_input == "spike":
            u_k_pool_t = k[t].float().mean(dim=-2)
        else:
            u_k_pool_t = k[t].float().mean(dim=(-2, -1)).unsqueeze(-1).expand(-1, -1, D)
        if module.gate_input_norm is not None and module.gamma_input != "rate":
            u_k_pool_t = module.gate_input_norm(u_k_pool_t)
        r_pool_t = k[t].float().mean(dim=(-2, -1))

        kv_t = k[t].transpose(-2, -1) @ v[t]

        if module.use_gamma_gate:
            u_gamma = alpha_gamma * u_gamma + u_k_pool_t
            s_gamma = module._gate_surrogate(u_gamma - V_gamma)
            s_gamma_reset = s_gamma.detach() if module._k_detach_reset else s_gamma
            u_gamma = u_gamma - s_gamma_reset * V_gamma
        else:
            s_gamma = k.new_zeros(B, H, D)

        if module.use_beta_gate:
            u_beta = alpha_beta * u_beta + r_pool_t
            s_beta = module._gate_surrogate(u_beta - V_beta)
            s_beta_reset = s_beta.detach() if module._k_detach_reset else s_beta
            u_beta = u_beta - s_beta_reset * V_beta
        else:
            s_beta = k.new_ones(B, H)

        gamma_fire_sum = gamma_fire_sum + s_gamma.detach().float().mean()
        beta_fire_sum = beta_fire_sum + s_beta.detach().float().mean()

        decay_mask = s_gamma * module.shift_scale
        alpha_eff = 1.0 - decay_mask
        S_prev = (
            alpha_eff.unsqueeze(-1) * S_prev
            + write_scale * s_beta.unsqueeze(-1).unsqueeze(-1) * kv_t
        )
        outputs.append((q[t] @ S_prev) * module.scale)

    with torch.no_grad():
        module.last_gamma_rate.copy_(gamma_fire_sum / float(T))
        module.last_beta_rate.copy_(beta_fire_sum / float(T))
    return torch.stack(outputs, dim=0)


def _streaming_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    if self.training or (torch.is_grad_enabled() and x.requires_grad):
        return self._nanosnn_orig_forward(x)
    attn_type = str(getattr(self, "attn_type", "")).lower()
    if attn_type not in _GATED_ATTN_TYPES:
        return self._nanosnn_orig_forward(x)

    T, B, _, _ = x.shape
    q, v = self._project_qv(x)
    k, k_membrane = self._project_k(x)

    if bool(getattr(self, "_nanosnn_streaming_attn_triton", False)) and x.device.type == "npu":
        out = _triton_forward(self, q, k, v, k_membrane, attn_type)
        return _project_out(self, out, T, B)

    if attn_type == "c0_sdla":
        out = _forward_c0(self, q, k, v)
    elif attn_type == "c1_lowrank":
        out = _forward_c1(self, q, k, v)
    elif attn_type == "c2_oneminusk":
        out = _forward_c2(self, q, k, v)
    elif attn_type == "c3_mga":
        if k_membrane is None:
            return self._nanosnn_orig_forward(x)
        out = _forward_c3(self, q, k, v, k_membrane)
    else:
        return self._nanosnn_orig_forward(x)

    return _project_out(self, out, T, B)


def patch_gated_attention_streaming_inference(module: torch.nn.Module, *, use_triton: bool = False) -> int:
    """Patch C0/C1/C2/C3 attention to avoid materializing full S_seq in eval.

    This is a forward-only memory optimization. It keeps the module class and
    state_dict unchanged, and falls back to the original forward during training.
    """
    patched = 0
    for submodule in module.modules():
        attn_type = str(getattr(submodule, "attn_type", "")).lower()
        if attn_type not in _GATED_ATTN_TYPES:
            continue
        if getattr(submodule, "_nanosnn_streaming_attn_patched", False):
            continue
        submodule._nanosnn_orig_forward = submodule.forward
        submodule.forward = types.MethodType(_streaming_forward, submodule)
        submodule._nanosnn_streaming_attn_triton = bool(use_triton)
        submodule._nanosnn_streaming_attn_patched = True
        patched += 1
    return patched


def unpatch_gated_attention_streaming(module: torch.nn.Module) -> int:
    restored = 0
    for submodule in module.modules():
        if not getattr(submodule, "_nanosnn_streaming_attn_patched", False):
            continue
        submodule.forward = submodule._nanosnn_orig_forward
        delattr(submodule, "_nanosnn_orig_forward")
        delattr(submodule, "_nanosnn_streaming_attn_triton")
        delattr(submodule, "_nanosnn_streaming_attn_patched")
        restored += 1
    return restored
