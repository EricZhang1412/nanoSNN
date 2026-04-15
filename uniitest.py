import math
import unittest
from types import SimpleNamespace

import torch

from models.common import tl_neuron_ops as ops


def _make_config(
    neuron_type: str = "lif",
    surrogate: str = "atan",
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: float | None = 0.0,
    detach_reset: bool = True,
    neuron_output: str = "spike",
    surrogate_alpha: float | None = None,
):
    return SimpleNamespace(
        neuron_type=neuron_type,
        surrogate=surrogate,
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        neuron_output=neuron_output,
        surrogate_alpha=surrogate_alpha,
    )


def _plif_w_from_tau(tau: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(-math.log(tau - 1.0), device=device, dtype=dtype)


class TritonNeuronSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def assertTensorClose(self, a: torch.Tensor, b: torch.Tensor, atol=1e-5, rtol=1e-4):
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(
            torch.allclose(a, b, atol=atol, rtol=rtol),
            msg=f"max_abs_diff={(a - b).abs().max().item()}",
        )

    def test_public_api_smoke_cpu(self):
        x = torch.randn(5, 2, 3, requires_grad=True)

        for neuron_type in ("if", "lif", "plif"):
            with self.subTest(neuron_type=neuron_type, output_mode="spike"):
                cfg = _make_config(neuron_type=neuron_type, neuron_output="spike", tau=2.3)
                node = ops.build_triton_neuron(cfg, step_mode="m", output_mode="spike")
                spike = node(x)
                self.assertEqual(spike.shape, x.shape)
                loss = spike.sum()
                loss.backward(retain_graph=True)

            with self.subTest(neuron_type=neuron_type, output_mode="psp"):
                cfg = _make_config(neuron_type=neuron_type, neuron_output="psp", tau=2.3)
                node = ops.build_triton_neuron(cfg, step_mode="m", output_mode="psp")
                v_seq = node(x)
                self.assertEqual(v_seq.shape, x.shape)
                loss = v_seq.sum()
                loss.backward(retain_graph=True)

            with self.subTest(neuron_type=neuron_type, output_mode="dual"):
                cfg = _make_config(neuron_type=neuron_type, neuron_output="dual", tau=2.3)
                node = ops.build_triton_neuron(cfg, step_mode="m", output_mode="dual")
                out = node(x)
                self.assertEqual(out.spike.shape, x.shape)
                self.assertEqual(out.v_seq.shape, x.shape)
                loss = out.spike.sum() + out.v_seq.sum()
                loss.backward(retain_graph=True)

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_single_step_psp_smoke_cpu(self):
        x = torch.randn(4, 6, requires_grad=True)
        cfg = _make_config(neuron_type="lif", neuron_output="psp", tau=2.5, v_reset=None)
        node = ops.build_triton_neuron(cfg, step_mode="s", output_mode="psp")
        v = node(x)
        self.assertEqual(v.shape, x.shape)
        v.sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(torch.cuda.is_available() and ops.triton is not None, "CUDA + Triton required")
class TritonKernelParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.device = torch.device("cuda")
        self.dtype = torch.float32

    def assertTensorClose(self, a: torch.Tensor, b: torch.Tensor, atol=1e-5, rtol=1e-4):
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(
            torch.allclose(a, b, atol=atol, rtol=rtol),
            msg=f"max_abs_diff={(a - b).abs().max().item()}",
        )

    def _case_param(self, neuron_type: str, tau: float) -> torch.Tensor:
        if neuron_type == "if":
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if neuron_type == "lif":
            return torch.tensor(tau, device=self.device, dtype=self.dtype)
        return _plif_w_from_tau(tau, self.device, self.dtype)

    def _run_forward_case(self, neuron_type: str, v_reset: float | None, surrogate: str):
        T, N = 9, 257
        x_flat = torch.randn(T, N, device=self.device, dtype=self.dtype)
        v_init = torch.randn(N, device=self.device, dtype=self.dtype)
        tau = 2.4
        param = self._case_param(neuron_type, tau)
        neuron_kind = ops._NEURON_KIND[neuron_type]

        spike_ref, v_ref = ops._forward_torch_flat(
            x_flat=x_flat,
            v_init=v_init,
            param=param,
            v_threshold=1.0,
            v_reset=v_reset,
            neuron_kind=neuron_kind,
        )
        spike_tri, v_tri = ops._forward_triton_flat(
            x_flat=x_flat,
            v_init=v_init,
            param=param,
            v_threshold=1.0,
            v_reset=v_reset,
            neuron_kind=neuron_kind,
        )

        self.assertTensorClose(spike_tri, spike_ref)
        self.assertTensorClose(v_tri, v_ref)

    def _run_backward_case(self, neuron_type: str, v_reset: float | None, surrogate: str, detach_reset: bool):
        T, N = 9, 257
        x_flat = torch.randn(T, N, device=self.device, dtype=self.dtype)
        v_init = torch.randn(N, device=self.device, dtype=self.dtype)
        grad_spike = torch.randn(T, N, device=self.device, dtype=self.dtype)
        grad_v_seq = torch.randn(T, N, device=self.device, dtype=self.dtype)
        tau = 2.4
        param = self._case_param(neuron_type, tau)
        neuron_kind = ops._NEURON_KIND[neuron_type]
        surrogate_kind = ops._SURROGATE_KIND[surrogate]
        need_param_grad = neuron_type == "plif"

        _, v_seq = ops._forward_torch_flat(
            x_flat=x_flat,
            v_init=v_init,
            param=param,
            v_threshold=1.0,
            v_reset=v_reset,
            neuron_kind=neuron_kind,
        )

        grad_x_ref, grad_param_ref = ops._backward_torch_flat(
            x_flat=x_flat,
            v_init=v_init,
            v_seq=v_seq,
            param=param,
            grad_spike=grad_spike,
            grad_v_seq=grad_v_seq,
            v_threshold=1.0,
            v_reset=v_reset,
            detach_reset=detach_reset,
            neuron_kind=neuron_kind,
            surrogate_kind=surrogate_kind,
            surrogate_alpha=ops._default_alpha(surrogate),
            need_param_grad=need_param_grad,
        )

        grad_x_tri, grad_param_tri = ops._backward_triton_flat(
            x_flat=x_flat,
            v_init=v_init,
            v_seq=v_seq,
            param=param,
            grad_spike=grad_spike,
            grad_v_seq=grad_v_seq,
            v_threshold=1.0,
            v_reset=v_reset,
            detach_reset=detach_reset,
            neuron_kind=neuron_kind,
            surrogate_kind=surrogate_kind,
            surrogate_alpha=ops._default_alpha(surrogate),
            need_param_grad=need_param_grad,
        )

        self.assertTensorClose(grad_x_tri, grad_x_ref, atol=2e-5, rtol=2e-4)
        if need_param_grad:
            self.assertTensorClose(grad_param_tri.reshape(1), grad_param_ref.reshape(1), atol=2e-5, rtol=2e-4)

    def test_forward_kernel_matches_torch_reference(self):
        cases = [
            ("if", 0.0, "atan"),
            ("lif", None, "atan"),
            ("lif", 0.2, "sigmoid"),
            ("plif", None, "atan"),
            ("plif", 0.1, "sigmoid"),
        ]
        for neuron_type, v_reset, surrogate in cases:
            with self.subTest(neuron_type=neuron_type, v_reset=v_reset, surrogate=surrogate):
                self._run_forward_case(neuron_type, v_reset, surrogate)

    def test_backward_kernel_matches_torch_reference(self):
        cases = [
            ("if", 0.0, "atan", True),
            ("lif", None, "atan", True),
            ("lif", 0.2, "sigmoid", False),
            ("plif", None, "atan", True),
            ("plif", 0.1, "sigmoid", False),
        ]
        for neuron_type, v_reset, surrogate, detach_reset in cases:
            with self.subTest(
                neuron_type=neuron_type,
                v_reset=v_reset,
                surrogate=surrogate,
                detach_reset=detach_reset,
            ):
                self._run_backward_case(neuron_type, v_reset, surrogate, detach_reset)

    def test_public_api_dual_matches_forced_torch_path(self):
        x = torch.randn(7, 3, 5, device=self.device, dtype=self.dtype, requires_grad=True)
        cfg = _make_config(
            neuron_type="plif",
            surrogate="atan",
            tau=2.6,
            v_threshold=1.0,
            v_reset=0.1,
            detach_reset=False,
            neuron_output="dual",
        )

        node_triton = ops.build_triton_neuron(cfg, step_mode="m", output_mode="dual").to(self.device)
        out_triton = node_triton(x)
        loss_triton = out_triton.spike.sum() + out_triton.v_seq.sum()
        loss_triton.backward()
        grad_triton = x.grad.detach().clone()

        x_ref = x.detach().clone().requires_grad_(True)
        original = ops._can_use_triton
        try:
            ops._can_use_triton = lambda _: False
            node_torch = ops.build_triton_neuron(cfg, step_mode="m", output_mode="dual").to(self.device)
            out_torch = node_torch(x_ref)
            loss_torch = out_torch.spike.sum() + out_torch.v_seq.sum()
            loss_torch.backward()
        finally:
            ops._can_use_triton = original

        self.assertTensorClose(out_triton.spike, out_torch.spike)
        self.assertTensorClose(out_triton.v_seq, out_torch.v_seq, atol=2e-5, rtol=2e-4)
        self.assertTensorClose(grad_triton, x_ref.grad, atol=2e-5, rtol=2e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)