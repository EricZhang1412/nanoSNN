# test_sdbgla_fwd.py
import torch
from mga_fwd_triton_kernel import sdbgla_fwd
from mga_bwd_triton_kernel import sdbgla_bwd_ds, sdbgla_bwd_qkvsg
from mga_torch_infer import sdbgla_reference


def test_fwd_numerical(dtype=torch.bfloat16, atol=1e-2, rtol=1e-2):
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    
    # Spike inputs (0/1)
    q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).to(dtype)
    
    delta = 0.5
    scale = 1.0 / (D ** 0.5)
    
    # Reference
    o_ref = sdbgla_reference(q, k, v, sg, delta, scale)
    
    # Triton
    o_tri = sdbgla_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), sg.contiguous(),
        delta, scale,
    )
    
    # Compare
    diff = (o_ref - o_tri).abs()
    print(f"max diff:  {diff.max().item():.6f}")
    print(f"mean diff: {diff.mean().item():.6f}")
    print(f"ref norm:  {o_ref.abs().mean().item():.6f}")
    
    assert torch.allclose(o_ref, o_tri, atol=atol, rtol=rtol), \
        f"FAIL: max diff = {diff.max()}"
    print("✓ Forward numerical test passed")


def test_fwd_edge_cases():
    torch.manual_seed(1)
    cases = [
        (4, 1, 1, 32, 32),     # minimum
        (1, 2, 4, 64, 32),     # T=1
        (4, 2, 4, 100, 32),    # N not divisible by BLOCK_N
        (4, 1, 1, 3136, 32),   # real ImageNet size
        (4, 2, 4, 128, 64),    # D=64
    ]
    for T, B, H, N, D in cases:
        q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(torch.bfloat16)
        k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(torch.bfloat16)
        v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(torch.bfloat16)
        sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).to(torch.bfloat16)
        
        o_ref = sdbgla_reference(q, k, v, sg, 0.5, 1.0 / (D ** 0.5))
        o_tri = sdbgla_fwd(q.contiguous(), k.contiguous(), v.contiguous(),
                           sg.contiguous(), 0.5, 1.0 / (D ** 0.5))
        
        diff = (o_ref - o_tri).abs().max().item()
        print(f"T={T} B={B} H={H} N={N} D={D}: max diff = {diff:.4f}")
        assert diff < 0.05, f"FAIL at {T},{B},{H},{N},{D}"
    print("✓ Edge cases passed")
    
def test_fwd_numerical_fp32(atol=1e-4, rtol=1e-4):
    """Ground truth correctness test in fp32."""
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    
    q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float()
    k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float()
    v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float()
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).float()
    
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    o_ref = sdbgla_reference(q, k, v, sg, delta, scale)
    o_tri = sdbgla_fwd(q.contiguous(), k.contiguous(), v.contiguous(),
                       sg.contiguous(), delta, scale)
    
    diff = (o_ref - o_tri).abs()
    print(f"[fp32] max diff:  {diff.max().item():.2e}")
    print(f"[fp32] mean diff: {diff.mean().item():.2e}")
    assert torch.allclose(o_ref, o_tri, atol=atol, rtol=rtol)
    print("✓ FP32 forward numerical test passed")

def test_fwd_numerical_bf16():
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    dtype = torch.bfloat16
    
    q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).to(dtype)
    
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    o_ref = sdbgla_reference(q, k, v, sg, delta, scale)
    o_tri = sdbgla_fwd(q.contiguous(), k.contiguous(), v.contiguous(),
                       sg.contiguous(), delta, scale)
    
    diff = (o_ref.float() - o_tri.float()).abs()
    ref_abs = o_ref.abs().float()
    ref_mean = ref_abs.mean()
    
    rel_mean = (diff.mean() / ref_mean).item()
    rel_max = (diff.max() / ref_mean).item()
    
    print(f"[bf16] ref mean:     {ref_mean.item():.4f}")
    print(f"[bf16] mean diff:    {diff.mean().item():.4f}  ({rel_mean:.2%})")
    print(f"[bf16] max diff:     {diff.max().item():.4f}  ({rel_max:.2%})")
    
    # bf16 容忍：相对 mean error < 1%, relative max error < 5%
    assert rel_mean < 0.01, f"mean rel error {rel_mean:.2%} too high"
    assert rel_max < 0.05, f"max rel error {rel_max:.2%} too high"
    print("✓ BF16 forward numerical test passed")
    
def test_fwd_in_autograd():
    """Sanity check: forward runs in autograd context, outputs are valid."""
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    dtype = torch.bfloat16
    
    # 1) Build inputs as leaf tensors with requires_grad
    #    (detach to make them leaves, then flip requires_grad on)
    q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).to(dtype).detach().requires_grad_(True)
    
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    # 2) Forward — under no_grad since we don't have backward yet.
    #    This simulates inference / benchmarking path.
    with torch.no_grad():
        o = sdbgla_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), sg.contiguous(),
            delta, scale,
        )
    
    # 3) Sanity checks on the output
    assert o.shape == (T, B, H, N, D), f"shape mismatch: got {o.shape}"
    assert o.dtype == dtype, f"dtype mismatch: got {o.dtype}, expected {dtype}"
    assert torch.isfinite(o).all().item(), "output contains NaN or Inf"
    
    print(f"Output shape:  {tuple(o.shape)}")
    print(f"Output dtype:  {o.dtype}")
    print(f"Output finite: {torch.isfinite(o).all().item()}")
    print(f"Output range:  [{o.min().item():.2f}, {o.max().item():.2f}]")
    print(f"Output |mean|: {o.abs().float().mean().item():.4f}")
    print("✓ Forward-in-autograd (no_grad path) passed")


def test_fwd_does_not_leak_graph():
    """Kernel is a detached op — it shouldn't keep leaf tensors' grad paths alive."""
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    dtype = torch.bfloat16
    
    q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).to(dtype).detach().requires_grad_(True)
    
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    with torch.no_grad():
        o = sdbgla_fwd(q.contiguous(), k.contiguous(), v.contiguous(),
                       sg.contiguous(), delta, scale)
    
    # Triton kernel output should have no grad_fn (we ran under no_grad)
    assert o.requires_grad is False, (
        "Output requires grad even under no_grad — did the kernel somehow "
        "attach to the autograd graph?"
    )
    assert o.grad_fn is None, f"Output has grad_fn: {o.grad_fn}"
    print("✓ Kernel output is correctly detached under no_grad")


def test_fwd_in_autograd_via_function():
    """
    If you plan to use it in training, you need an autograd.Function wrapper.
    Here we test a stub wrapper that falls back to the PyTorch reference for bwd.
    This confirms: (fwd = Triton) + (bwd = PyTorch autograd) works end-to-end.
    """
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    dtype = torch.bfloat16
    
    class _SDBGLAFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, sg, delta, scale):
            ctx.save_for_backward(q, k, v, sg)
            ctx.delta = delta
            ctx.scale = scale
            return sdbgla_fwd(q.contiguous(), k.contiguous(),
                              v.contiguous(), sg.contiguous(), delta, scale)
        
        @staticmethod
        def backward(ctx, grad_output):
            # Recompute via PyTorch reference with autograd
            q, k, v, sg = ctx.saved_tensors
            with torch.enable_grad():
                q_ = q.detach().float().requires_grad_(q.requires_grad)
                k_ = k.detach().float().requires_grad_(k.requires_grad)
                v_ = v.detach().float().requires_grad_(v.requires_grad)
                sg_ = sg.detach().float().requires_grad_(sg.requires_grad)
                
                # Inline autograd-friendly reference (mirror of sdbgla_reference)
                S = torch.zeros(q_.shape[1], q_.shape[2], q_.shape[4], q_.shape[4],
                                dtype=torch.float32, device=q_.device)
                outs = []
                for t in range(q_.shape[0]):
                    KV = torch.einsum("bhnd,bhne->bhde", k_[t], v_[t])
                    decay = 1.0 - sg_[t] * ctx.delta
                    S = S * decay.unsqueeze(-1) + KV
                    outs.append(torch.einsum("bhnk,bhkv->bhnv", q_[t], S) * ctx.scale)
                o = torch.stack(outs).to(grad_output.dtype)
                
                grads = torch.autograd.grad(
                    o, [q_, k_, v_, sg_],
                    grad_outputs=grad_output,
                    allow_unused=True,
                )
            
            def _match(g, ref):
                if g is None:
                    return None
                return g.to(ref.dtype)
            
            return (_match(grads[0], q), _match(grads[1], k),
                    _match(grads[2], v), _match(grads[3], sg),
                    None, None)
    
    q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype).detach().requires_grad_(True)
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).to(dtype).detach().requires_grad_(True)
    
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    # Forward through the autograd.Function
    o = _SDBGLAFn.apply(q, k, v, sg, delta, scale)
    
    assert o.requires_grad is True, "Output should require grad"
    assert o.grad_fn is not None, "Output should have grad_fn"
    
    # Backward
    loss = o.sum()
    loss.backward()
    
    # Check all leaves got gradients
    for name, t in [("q", q), ("k", k), ("v", v), ("sg", sg)]:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all().item(), f"{name}.grad has NaN/Inf"
        print(f"  {name}.grad: shape={tuple(t.grad.shape)}, "
              f"|mean|={t.grad.abs().float().mean().item():.4e}, "
              f"|max|={t.grad.abs().float().max().item():.4e}")
    
    print("✓ Autograd.Function (Triton fwd + PyTorch bwd) works end-to-end")

def benchmark_fwd():
    """Compare fused Triton fwd vs the PyTorch reference path."""
    import time
    
    # ImageNet-like size
    T, B, H, N, D = 4, 32, 12, 3136, 32
    dtype = torch.bfloat16
    
    q = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    k = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    v = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).to(dtype)
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).to(dtype)
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    # Warmup
    for _ in range(3):
        _ = sdbgla_fwd(q.contiguous(), k.contiguous(), v.contiguous(), sg.contiguous(), delta, scale)
    torch.cuda.synchronize()
    
    # Triton fwd timing
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(20):
        o = sdbgla_fwd(q.contiguous(), k.contiguous(), v.contiguous(), sg.contiguous(), delta, scale)
    torch.cuda.synchronize()
    t_tri = (time.perf_counter() - t0) / 20
    mem_tri = torch.cuda.max_memory_allocated() / 1e9
    
    # PyTorch reference timing (the recurrent form, like your _forward_recurrent)
    def pytorch_path(q, k, v, sg):
        kv = torch.einsum("TBHND,TBHNE->TBHDE", k.float(), v.float())
        S = torch.zeros_like(kv[0])
        outs = []
        for t in range(T):
            gate = sg[t].float().unsqueeze(-1)
            S = S * (1.0 - gate * delta) + kv[t]
            outs.append(S)
        S_all = torch.stack(outs)
        return torch.einsum("TBHvk,TBHNk->TBHNv", S_all, q.float()) * scale
    
    for _ in range(3):
        _ = pytorch_path(q, k, v, sg)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(20):
        o = pytorch_path(q, k, v, sg)
    torch.cuda.synchronize()
    t_py = (time.perf_counter() - t0) / 20
    mem_py = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"Triton fwd:  {t_tri*1000:.2f} ms,  peak mem {mem_tri:.2f} GB")
    print(f"PyTorch fwd: {t_py*1000:.2f} ms,  peak mem {mem_py:.2f} GB")
    print(f"Speedup: {t_py/t_tri:.2f}x,  Memory savings: {mem_py-mem_tri:.2f} GB")
 
def test_bwd_ds_kernel():
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False 
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    
    # Inputs must require grad so S_t nodes end up in autograd graph
    q  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
    k  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
    v  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).float().detach().requires_grad_()
    do = torch.randn(T, B, H, N, D, device='cuda')
    
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    # Reference: autograd graph with retained S_t grads
    S = torch.zeros(B, H, D, D, device='cuda', dtype=torch.float32)
    S_list = []
    o_list = []
    for t in range(T):
        KV = torch.einsum("bhnd,bhne->bhde", k[t], v[t])
        decay = 1.0 - sg[t] * delta
        S = S * decay.unsqueeze(-1) + KV
        S.retain_grad()
        S_list.append(S)
        o_list.append(torch.einsum("bhnk,bhkv->bhnv", q[t], S) * scale)
    o_ref = torch.stack(o_list)
    o_ref.backward(do)
    dS_ref = torch.stack([s.grad for s in S_list])
    
    # Triton (use .detach() to feed raw tensors — kernel doesn't need grad)
    dS_tri = sdbgla_bwd_ds(
        q.detach().contiguous(),
        do.contiguous(),
        sg.detach().contiguous(),
        delta, scale, precision="ieee",
    )
    
    diff = (dS_ref - dS_tri).abs()
    print(f"[dS] shape:     {tuple(dS_tri.shape)}")
    print(f"[dS] ref mean:  {dS_ref.abs().mean().item():.4e}")
    print(f"[dS] max diff:  {diff.max().item():.2e}")
    print(f"[dS] mean diff: {diff.mean().item():.2e}")
    
    for t in range(T):
        d_t = (dS_ref[t] - dS_tri[t]).abs()
        print(f"  t={t}: max={d_t.max().item():.2e}, "
              f"ref_abs_mean={dS_ref[t].abs().mean().item():.4f}")
    
    assert diff.max().item() < 1e-4, \
        f"dS kernel numerical check FAILED (max diff = {diff.max()})"
    print("✓ dS kernel (Kernel A) passed")


def test_bwd_ds_edge_cases():
    torch.manual_seed(1)
    cases = [
        (1, 2, 4, 64, 32),
        (4, 1, 1, 32, 32),
        (4, 2, 4, 100, 32),
        (4, 1, 1, 3136, 32),
        (4, 2, 4, 128, 64),
    ]
    for T, B, H, N, D in cases:
        q  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
        k  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
        v  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
        sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).float().detach().requires_grad_()
        do = torch.randn(T, B, H, N, D, device='cuda')
        scale = 1.0 / (D ** 0.5)
        
        S = torch.zeros(B, H, D, D, device='cuda', dtype=torch.float32)
        S_list = []
        o_list = []
        for t in range(T):
            KV = torch.einsum("bhnd,bhne->bhde", k[t], v[t])
            S = S * (1.0 - sg[t] * 0.5).unsqueeze(-1) + KV
            S.retain_grad()
            S_list.append(S)
            o_list.append(torch.einsum("bhnk,bhkv->bhnv", q[t], S) * scale)
        o_ref = torch.stack(o_list)
        o_ref.backward(do)
        dS_ref = torch.stack([s.grad for s in S_list])
        
        dS_tri = sdbgla_bwd_ds(q.detach().contiguous(), do.contiguous(),
                               sg.detach().contiguous(), 0.5, scale,
                               precision="ieee")
        
        diff = (dS_ref - dS_tri).abs().max().item()
        print(f"[dS] T={T} B={B} H={H} N={N} D={D}: max diff = {diff:.2e}")
        assert diff < 1e-4, f"FAIL at {(T,B,H,N,D)}: {diff}"
    print("✓ dS kernel edge cases passed")
    
def test_bwd_qkvsg_kernel():
    """Verify Kernel B: dQ, dK, dV, dsg correctness."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    torch.manual_seed(0)
    T, B, H, N, D = 4, 2, 4, 128, 32
    
    q  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
    k  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
    v  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
    sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).float().detach().requires_grad_()
    do = torch.randn(T, B, H, N, D, device='cuda')
    
    delta, scale = 0.5, 1.0 / (D ** 0.5)
    
    # Reference: build autograd graph, compute grads for Q, K, V, sg.
    # Also extract S_all and dS_all from the graph for Triton.
    S = torch.zeros(B, H, D, D, device='cuda', dtype=torch.float32)
    S_list = []
    o_list = []
    for t in range(T):
        KV = torch.einsum("bhnd,bhne->bhde", k[t], v[t])
        decay = 1.0 - sg[t] * delta
        S = S * decay.unsqueeze(-1) + KV
        S.retain_grad()
        S_list.append(S)
        o_list.append(torch.einsum("bhnk,bhkv->bhnv", q[t], S) * scale)
    o_ref = torch.stack(o_list)
    o_ref.backward(do)
    
    dq_ref = q.grad
    dk_ref = k.grad
    dv_ref = v.grad
    dsg_ref = sg.grad
    
    # S_all and dS_all from autograd (used as Kernel B inputs)
    S_all_ref = torch.stack([s.detach() for s in S_list])           # [T, B, H, D, D]
    dS_all_ref = torch.stack([s.grad for s in S_list])              # [T, B, H, D, D]
    
    # Triton
    dq_tri, dk_tri, dv_tri, dsg_tri = sdbgla_bwd_qkvsg(
        q.detach().contiguous(), k.detach().contiguous(),
        v.detach().contiguous(), sg.detach().contiguous(),
        S_all_ref.contiguous(), dS_all_ref.contiguous(),
        do.contiguous(),
        delta, scale, precision="ieee",
    )
    
    def report(name, ref, tri):
        diff = (ref - tri).abs()
        print(f"[{name}] shape={tuple(tri.shape)}, ref_mean={ref.abs().mean():.4e}, "
              f"max_diff={diff.max().item():.2e}, mean_diff={diff.mean().item():.2e}")
        return diff.max().item()
    
    max_dq  = report("dQ",  dq_ref,  dq_tri)
    max_dk  = report("dK",  dk_ref,  dk_tri)
    max_dv  = report("dV",  dv_ref,  dv_tri)
    max_dsg = report("dsg", dsg_ref, dsg_tri)
    
    # Per-timestep dsg diagnostic (catches time-index bugs)
    for t in range(T):
        d_t = (dsg_ref[t] - dsg_tri[t]).abs()
        print(f"  dsg t={t}: max={d_t.max().item():.2e}, "
              f"ref_mean={dsg_ref[t].abs().mean().item():.4e}")
    
    assert max_dq  < 1e-4, f"dQ  FAIL: {max_dq}"
    assert max_dk  < 1e-4, f"dK  FAIL: {max_dk}"
    assert max_dv  < 1e-4, f"dV  FAIL: {max_dv}"
    assert max_dsg < 1e-4, f"dsg FAIL: {max_dsg}"
    print("✓ Kernel B (dQ, dK, dV, dsg) passed")


def test_bwd_qkvsg_edge_cases():
    """Edge cases for Kernel B."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    torch.manual_seed(1)
    cases = [
        (1, 2, 4, 64, 32),       # T=1: dsg_0 = 0 branch exclusively
        (4, 1, 1, 32, 32),       # tiny
        (4, 2, 4, 100, 32),      # N % BLOCK_N != 0
        (4, 1, 1, 3136, 32),     # ImageNet size
        (4, 2, 4, 128, 64),      # D=64
    ]
    for T, B, H, N, D in cases:
        q  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
        k  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
        v  = (torch.rand(T, B, H, N, D, device='cuda') > 0.7).float().detach().requires_grad_()
        sg = (torch.rand(T, B, H, D, device='cuda') > 0.5).float().detach().requires_grad_()
        do = torch.randn(T, B, H, N, D, device='cuda')
        scale = 1.0 / (D ** 0.5)
        
        S = torch.zeros(B, H, D, D, device='cuda', dtype=torch.float32)
        S_list = []
        o_list = []
        for t in range(T):
            KV = torch.einsum("bhnd,bhne->bhde", k[t], v[t])
            S = S * (1.0 - sg[t] * 0.5).unsqueeze(-1) + KV
            S.retain_grad()
            S_list.append(S)
            o_list.append(torch.einsum("bhnk,bhkv->bhnv", q[t], S) * scale)
        o_ref = torch.stack(o_list)
        o_ref.backward(do)
        
        S_all = torch.stack([s.detach() for s in S_list]).contiguous()
        dS_all = torch.stack([s.grad for s in S_list]).contiguous()
        
        dq_tri, dk_tri, dv_tri, dsg_tri = sdbgla_bwd_qkvsg(
            q.detach().contiguous(), k.detach().contiguous(),
            v.detach().contiguous(), sg.detach().contiguous(),
            S_all, dS_all, do.contiguous(),
            0.5, scale, precision="ieee",
        )
        
        m_dq  = (q.grad  - dq_tri).abs().max().item()
        m_dk  = (k.grad  - dk_tri).abs().max().item()
        m_dv  = (v.grad  - dv_tri).abs().max().item()
        m_dsg = (sg.grad - dsg_tri).abs().max().item()
        print(f"T={T} B={B} H={H} N={N} D={D}: "
              f"dQ={m_dq:.1e} dK={m_dk:.1e} dV={m_dv:.1e} dsg={m_dsg:.1e}")
        assert max(m_dq, m_dk, m_dv, m_dsg) < 1e-4, f"FAIL at {(T,B,H,N,D)}"
    print("✓ Kernel B edge cases passed")
    
    
if __name__ == "__main__":
    # test_fwd_numerical_fp32()
    # test_fwd_numerical_bf16()
    # test_fwd_in_autograd()
    # test_fwd_does_not_leak_graph()
    # test_fwd_in_autograd_via_function()
    # benchmark_fwd()
    
    test_bwd_ds_kernel()
    test_bwd_ds_edge_cases()
    test_bwd_qkvsg_kernel()
    test_bwd_qkvsg_edge_cases()
    