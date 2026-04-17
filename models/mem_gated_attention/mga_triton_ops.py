# models/mem_gated_attention/sdbgla_triton.py
# (append this to the file containing your kernels)

import torch
from .mga_fwd_triton_kernel import sdbgla_fwd
from .mga_bwd_triton_kernel import sdbgla_bwd_ds, sdbgla_bwd_qkvsg


class SDBGLAFunction(torch.autograd.Function):
    """
    Fully-fused SD-BGLA attention with Triton forward + backward kernels.
    
    Forward:  Triton fused kernel (from sdbgla_fwd), outputs O and S_all.
    Backward: Kernel A (dS_all) + Kernel B (dQ, dK, dV, dsg).
    """
    
    @staticmethod
    def forward(ctx, q, k, v, sg, delta, scale, precision):
        """
        Args:
            q, k, v: [T, B, H, N, D]  contiguous
            sg:      [T, B, H, D]      contiguous
            delta, scale: python floats
            precision: "ieee" | "tf32" | "tf32x3"  (for matmul in kernels)
        """
        # Forward produces both O and S_all; S_all is needed for backward.
        o, S_all = sdbgla_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), sg.contiguous(),
            delta, scale,
            precision=precision,
            return_S=True,
        )
        
        ctx.save_for_backward(q, k, v, sg, S_all)
        ctx.delta = delta
        ctx.scale = scale
        ctx.precision = precision
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, sg, S_all = ctx.saved_tensors
        delta, scale, precision = ctx.delta, ctx.scale, ctx.precision
        
        # Autograd can pass non-contiguous grad_output; saved tensors may also
        # become non-contiguous through view/expand operations. Force contiguous
        # on everything entering the Triton kernels.
        do     = grad_output.contiguous()
        q      = q.contiguous()
        k      = k.contiguous()
        v      = v.contiguous()
        sg     = sg.contiguous()
        S_all  = S_all.contiguous()
        
        dS_all = sdbgla_bwd_ds(q, do, sg, delta, scale, precision=precision)
        
        dq, dk, dv, dsg = sdbgla_bwd_qkvsg(
            q, k, v, sg, S_all, dS_all, do,
            delta, scale, precision=precision,
        )
        
        return dq, dk, dv, dsg, None, None, None


def sdbgla_attention(q, k, v, sg, delta: float, scale: float,
                     precision: str = "tf32") -> torch.Tensor:
    """User-facing API — drop-in replacement for the three einsum + recurrence steps.
    
    Args:
        q, k, v:  [T, B, H, N, D]   spike tensors (0/1 valued, any float dtype)
        sg:       [T, B, H, D]      gate spike tensor
        delta:    bit-shift decay (e.g. 0.5 for shift_k=1)
        scale:    attention scale (e.g. D ** -0.5)
        precision: matmul precision mode inside kernels
    
    Returns:
        o:        [T, B, H, N, D]   attention output, same dtype as q
    """
    return SDBGLAFunction.apply(q, k, v, sg, delta, scale, precision)