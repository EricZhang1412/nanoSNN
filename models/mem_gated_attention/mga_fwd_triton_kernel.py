# models/mem_gated_attention/sdbgla_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def _sdbgla_fwd_kernel(
    q_ptr, k_ptr, v_ptr, sg_ptr, o_ptr, s_all_ptr,
    s_qt, s_qb, s_qh, s_qn,
    s_sgt, s_sgb, s_sgh,
    s_sat, s_sab, s_sah, s_sad0,
    T, B, H, N,
    delta, scale,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_d = tl.arange(0, D)
    
    S = tl.zeros([D, D], dtype=tl.float32)
    
    for t in range(T):
        qkv_base = t * s_qt + b * s_qb + h * s_qh
        sg_base = t * s_sgt + b * s_sgb + h * s_sgh
        sa_base = t * s_sat + b * s_sab + h * s_sah
        
        # Phase 1: KV
        KV = tl.zeros([D, D], dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            k_ptrs = k_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            v_ptrs = v_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            KV += tl.dot(tl.trans(k_tile), v_tile, input_precision=PRECISION)
        
        # Phase 2: recurrence
        sg = tl.load(sg_ptr + sg_base + offs_d).to(tl.float32)
        decay = 1.0 - sg * delta
        S = S * decay[:, None] + KV
        
        # Always store S_t (cheap, 1.6 MB at ImageNet scale)
        s_ptrs = s_all_ptr + sa_base + offs_d[:, None] * s_sad0 + offs_d[None, :]
        tl.store(s_ptrs, S)
        
        # Phase 3: O
        S_cast = S.to(q_ptr.dtype.element_ty)
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            q_ptrs = q_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
            q_tile = tl.load(q_ptrs, mask=mask_n[:, None], other=0.0)
            o_tile = tl.dot(q_tile, S_cast, input_precision=PRECISION)
            o_tile = o_tile * scale
            o_ptrs = o_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
            tl.store(o_ptrs, o_tile.to(q_ptr.dtype.element_ty), mask=mask_n[:, None])


def sdbgla_fwd(q, k, v, s_gamma, delta: float, scale: float,
               precision: str = "tf32", return_S: bool = False):
    """
    Args:
        return_S: if True, returns (o, S_all); else returns only o (S_all discarded).
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert s_gamma.is_contiguous()
    T, B, H, N, D = q.shape
    assert D in (16, 32, 64, 128)
    
    o = torch.empty_like(q)
    S_all = torch.empty(T, B, H, D, D, dtype=torch.float32, device=q.device)
    
    s_qt, s_qb, s_qh, s_qn, _ = q.stride()
    s_sgt, s_sgb, s_sgh, _ = s_gamma.stride()
    s_sat, s_sab, s_sah, s_sad0, _ = S_all.stride()
    
    BLOCK_N = min(64, triton.next_power_of_2(N))
    grid = (B * H,)
    
    _sdbgla_fwd_kernel[grid](
        q, k, v, s_gamma, o, S_all,
        s_qt, s_qb, s_qh, s_qn,
        s_sgt, s_sgb, s_sgh,
        s_sat, s_sab, s_sah, s_sad0,
        T, B, H, N,
        float(delta), float(scale),
        D=D,
        BLOCK_N=BLOCK_N,
        PRECISION=precision,
        num_warps=4,
        num_stages=2,
    )
    
    return (o, S_all) if return_S else o