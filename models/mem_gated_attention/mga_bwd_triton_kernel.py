import torch
import triton
import triton.language as tl


@triton.jit
def _sdbgla_bwd_ds_kernel(
    # Pointers
    q_ptr, do_ptr, sg_ptr, ds_all_ptr,
    # Strides for Q, dO [T, B, H, N, D] (same strides, shared)
    s_qt, s_qb, s_qh, s_qn,
    # Strides for sg [T, B, H, D]
    s_sgt, s_sgb, s_sgh,
    # Strides for dS_all [T, B, H, D, D]
    s_dst, s_dsb, s_dsh, s_dsd0,
    # Dims
    T, B, H, N,
    # Scalars
    delta, scale,
    # Constexpr
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_d = tl.arange(0, D)
    
    # dS_carry: gradient flowing back from t+1 to t, lives in SRAM across T
    dS_carry = tl.zeros([D, D], dtype=tl.float32)
    
    for rev_t in range(T):
        t = T - 1 - rev_t
        
        qkv_base = t * s_qt + b * s_qb + h * s_qh
        sg_base  = t * s_sgt + b * s_sgb + h * s_sgh
        ds_base  = t * s_dst + b * s_dsb + h * s_dsh
        
        # ========================================================
        # Phase 1: direct contribution  dS_direct = scale * Q_t^T @ dO_t
        # ========================================================
        dS_direct = tl.zeros([D, D], dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            
            # Q_t[n, :]: [BLOCK_N, D]
            q_ptrs = q_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
            q_tile = tl.load(q_ptrs, mask=mask_n[:, None], other=0.0)
            
            # dO_t[n, :]: [BLOCK_N, D]
            do_ptrs = do_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
            do_tile = tl.load(do_ptrs, mask=mask_n[:, None], other=0.0)
            
            # Q^T @ dO: [D, BLOCK_N] @ [BLOCK_N, D] = [D, D]
            dS_direct += tl.dot(tl.trans(q_tile), do_tile,
                                input_precision=PRECISION)
        
        dS_direct = dS_direct * scale
        
        # ========================================================
        # Phase 2: merge dS_t = dS_direct + dS_carry
        # ========================================================
        dS_t = dS_direct + dS_carry
        
        # ========================================================
        # Phase 3: store dS_t to HBM
        # ========================================================
        ds_ptrs = ds_all_ptr + ds_base + \
                  offs_d[:, None] * s_dsd0 + offs_d[None, :]
        tl.store(ds_ptrs, dS_t)
        
        # ========================================================
        # Phase 4: prepare carry for next iteration (t-1)
        # dS_carry_for_next = dS_t * decay_t   (row-wise on k-axis)
        # decay_t[k] = 1 - sg_t[k] * delta
        # ========================================================
        sg_ptrs = sg_ptr + sg_base + offs_d
        sg = tl.load(sg_ptrs).to(tl.float32)
        decay_t = 1.0 - sg * delta                 # [D]
        dS_carry = dS_t * decay_t[:, None]          # [D, D], row-wise


def sdbgla_bwd_ds(q, do, sg, delta: float, scale: float,
                  precision: str = "ieee"):
    """
    Compute dS_all given Q, dO, sg.
    
    Args:
        q:   [T, B, H, N, D]  (same dtype as forward Q)
        do:  [T, B, H, N, D]  (upstream gradient, same dtype as Q)
        sg:  [T, B, H, D]     (gate spikes)
        delta, scale: scalars matching forward
        precision: "ieee" | "tf32" | "tf32x3"  — for Q^T @ dO matmul
    
    Returns:
        dS_all: [T, B, H, D, D]  fp32
    """
    assert q.is_contiguous() and do.is_contiguous() and sg.is_contiguous()
    T, B, H, N, D = q.shape
    assert do.shape == q.shape, f"dO shape {do.shape} != Q shape {q.shape}"
    assert sg.shape == (T, B, H, D), f"sg shape {sg.shape}"
    assert D in (16, 32, 64, 128)
    
    dS_all = torch.empty(T, B, H, D, D, dtype=torch.float32, device=q.device)
    
    s_qt, s_qb, s_qh, s_qn, _ = q.stride()
    s_sgt, s_sgb, s_sgh, _ = sg.stride()
    s_dst, s_dsb, s_dsh, s_dsd0, _ = dS_all.stride()
    
    BLOCK_N = min(64, triton.next_power_of_2(N))
    grid = (B * H,)
    
    _sdbgla_bwd_ds_kernel[grid](
        q, do, sg, dS_all,
        s_qt, s_qb, s_qh, s_qn,
        s_sgt, s_sgb, s_sgh,
        s_dst, s_dsb, s_dsh, s_dsd0,
        T, B, H, N,
        float(delta), float(scale),
        D=D,
        BLOCK_N=BLOCK_N,
        PRECISION=precision,
        num_warps=4,
        num_stages=2,
    )
    return dS_all

@triton.jit
def _sdbgla_bwd_qkvsg_kernel(
    # Pointers
    q_ptr, k_ptr, v_ptr, sg_ptr,
    s_all_ptr, ds_all_ptr,
    do_ptr,
    dq_ptr, dk_ptr, dv_ptr, dsg_ptr,
    # Strides for Q/K/V/dQ/dK/dV/dO [T, B, H, N, D]
    s_qt, s_qb, s_qh, s_qn,
    # Strides for sg/dsg [T, B, H, D]
    s_sgt, s_sgb, s_sgh,
    # Strides for S_all/dS_all [T, B, H, D, D]
    s_sat, s_sab, s_sah, s_sad0,
    # Dims
    T, B, H, N,
    # Scalars
    delta, scale,
    # Constexpr
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    # Decompose (t, b, h) from 1D pid
    t = pid // (B * H)
    bh = pid % (B * H)
    b = bh // H
    h = bh % H
    
    offs_d = tl.arange(0, D)
    
    # Base offsets
    qkv_base = t * s_qt + b * s_qb + h * s_qh
    sg_base  = t * s_sgt + b * s_sgb + h * s_sgh
    sa_base  = t * s_sat + b * s_sab + h * s_sah
    
    # =========================================================
    # Load S_t and dS_t (both [D, D], fp32) — needed for Phase 1-4
    # =========================================================
    s_t_ptrs = s_all_ptr + sa_base + \
               offs_d[:, None] * s_sad0 + offs_d[None, :]
    S_t = tl.load(s_t_ptrs)                                      # [D, D] fp32
    
    ds_t_ptrs = ds_all_ptr + sa_base + \
                offs_d[:, None] * s_sad0 + offs_d[None, :]
    dS_t = tl.load(ds_t_ptrs)                                    # [D, D] fp32
    
    # =========================================================
    # Phase 1: dQ_t = scale * dO_t @ S_t^T   →  [N, D]
    # Phase 2: dK_t = V_t   @ dS_t^T          →  [N, D]
    # Phase 3: dV_t = K_t   @ dS_t            →  [N, D]
    # All three iterate over N tiles.
    # =========================================================
    
    # Pre-cast S_t and dS_t to input dtype for matmul
    S_t_cast  = S_t.to(q_ptr.dtype.element_ty)                   # [D, D]
    St_T_cast = tl.trans(S_t_cast)                               # [D, D]
    dS_t_cast = dS_t.to(q_ptr.dtype.element_ty)                  # [D, D]
    dSt_T_cast = tl.trans(dS_t_cast)                             # [D, D]
    
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # ----- Phase 1: dQ -----
        do_ptrs = do_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
        do_tile = tl.load(do_ptrs, mask=mask_n[:, None], other=0.0)
        
        # dQ = scale * dO @ S^T :  [BLOCK_N, D] @ [D, D] = [BLOCK_N, D]
        dq_tile = tl.dot(do_tile, St_T_cast, input_precision=PRECISION)
        dq_tile = dq_tile * scale
        
        dq_ptrs = dq_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
        tl.store(dq_ptrs, dq_tile.to(q_ptr.dtype.element_ty),
                 mask=mask_n[:, None])
        
        # ----- Phase 2: dK -----
        v_ptrs = v_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # dK = V @ dS^T :  [BLOCK_N, D] @ [D, D] = [BLOCK_N, D]
        dk_tile = tl.dot(v_tile, dSt_T_cast, input_precision=PRECISION)
        
        dk_ptrs = dk_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
        tl.store(dk_ptrs, dk_tile.to(q_ptr.dtype.element_ty),
                 mask=mask_n[:, None])
        
        # ----- Phase 3: dV -----
        k_ptrs = k_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # dV = K @ dS :  [BLOCK_N, D] @ [D, D] = [BLOCK_N, D]
        dv_tile = tl.dot(k_tile, dS_t_cast, input_precision=PRECISION)
        
        dv_ptrs = dv_ptr + qkv_base + offs_n[:, None] * s_qn + offs_d[None, :]
        tl.store(dv_ptrs, dv_tile.to(q_ptr.dtype.element_ty),
                 mask=mask_n[:, None])
    
    # =========================================================
    # Phase 4: dsg_t = -delta * sum_v (dS_t * S_{t-1})
    # Per-row reduce over v axis of [D, D]
    # Edge: t=0 → S_{-1} = 0 → dsg_0 = 0
    # =========================================================
    if t == 0:
        dsg = tl.zeros([D], dtype=tl.float32)
    else:
        sa_prev_base = (t - 1) * s_sat + b * s_sab + h * s_sah
        s_prev_ptrs = s_all_ptr + sa_prev_base + \
                      offs_d[:, None] * s_sad0 + offs_d[None, :]
        S_prev = tl.load(s_prev_ptrs)                            # [D, D] fp32
        
        # element-wise × then reduce over v axis (axis=1)
        prod = dS_t * S_prev                                     # [D, D]
        dsg = tl.sum(prod, axis=1)                               # [D]
        dsg = -delta * dsg
    
    # Store dsg_t
    dsg_ptrs = dsg_ptr + sg_base + offs_d
    tl.store(dsg_ptrs, dsg.to(sg_ptr.dtype.element_ty))


def sdbgla_bwd_qkvsg(q, k, v, sg, S_all, dS_all, do,
                     delta: float, scale: float,
                     precision: str = "ieee"):
    """
    Compute dQ, dK, dV, dsg given forward tensors and dS_all.
    
    Args:
        q, k, v:  [T, B, H, N, D]
        sg:       [T, B, H, D]
        S_all:    [T, B, H, D, D]   fp32   (from forward)
        dS_all:   [T, B, H, D, D]   fp32   (from Kernel A)
        do:       [T, B, H, N, D]   same dtype as Q
        
    Returns:
        dq, dk, dv:  [T, B, H, N, D]  same dtype as Q
        dsg:         [T, B, H, D]     same dtype as sg
    """
    for x in (q, k, v, sg, S_all, dS_all, do):
        assert x.is_contiguous(), f"tensor not contiguous"
    T, B, H, N, D = q.shape
    assert S_all.shape == (T, B, H, D, D)
    assert dS_all.shape == (T, B, H, D, D)
    assert D in (16, 32, 64, 128)
    
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dsg = torch.empty_like(sg)
    
    s_qt, s_qb, s_qh, s_qn, _ = q.stride()
    s_sgt, s_sgb, s_sgh, _ = sg.stride()
    s_sat, s_sab, s_sah, s_sad0, _ = S_all.stride()
    
    BLOCK_N = min(64, triton.next_power_of_2(N))
    grid = (T * B * H,)
    
    _sdbgla_bwd_qkvsg_kernel[grid](
        q, k, v, sg,
        S_all, dS_all,
        do,
        dq, dk, dv, dsg,
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
    
    return dq, dk, dv, dsg