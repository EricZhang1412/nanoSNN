# models/mem_gated_attention/sdbgla_reference.py
import torch

@torch.no_grad()
def sdbgla_reference(q, k, v, s_gamma, delta, scale):
    """
    SD-BGLA reference forward (pure PyTorch, no autograd).
    
    Args:
        q, k, v: [T, B, H, N, D]  (spike, 0/1)
        s_gamma: [T, B, H, D]      (spike, 0/1)
        delta: float                (e.g. 0.5)
        scale: float                (e.g. 1/sqrt(D))
    Returns:
        o: [T, B, H, N, D]
    """
    T, B, H, N, D = q.shape
    S = torch.zeros(B, H, D, D, dtype=torch.float32, device=q.device)
    o = torch.empty(T, B, H, N, D, dtype=q.dtype, device=q.device)
    
    for t in range(T):
        # KV_t = K_t^T @ V_t  →  [B, H, D, D]
        KV = torch.einsum("bhnd,bhne->bhde", k[t].float(), v[t].float())
        
        # S_t = S_{t-1} * (1 - s_gamma_t * delta) + KV_t
        decay = 1.0 - s_gamma[t].float() * delta   # [B, H, D]
        S = S * decay.unsqueeze(-1) + KV             # [B, H, D, D]
        
        # O_t = Q_t @ S^T * scale  (S is [D_k, D_v], output dim is D_v)
        # o_t[b, h, n, v] = sum_k q[b, h, n, k] * S[b, h, k, v] * scale
        o[t] = torch.einsum("bhnk,bhkv->bhnv", q[t].float(), S).to(q.dtype) * scale
    
    return o