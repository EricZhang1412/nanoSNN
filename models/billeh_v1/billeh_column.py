import numpy as np
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


def sparse_mm_batch(sparse_w, dense_x):
    # sparse_w: [out, in], dense_x: [B, in]
    return torch.sparse.mm(sparse_w, dense_x.t()).t()


def make_sparse(indices, weights, shape, device):
    idx = torch.as_tensor(indices.T, dtype=torch.long, device=device)
    val = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        idx, val, size=shape, device=device, check_invariants=False
    ).coalesce()


class BillehColumnTorch(nn.Module):
    def __init__(
        self,
        network,
        input_population,
        bkg_weights,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
        max_delay=5,
        train_recurrent=True,
        train_input=True,
        train_bkg=False,
        use_dale_law=True,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dt = float(dt)
        self.use_dale_law = use_dale_law

        params = network["node_params"]
        self.n_neurons = int(network["n_nodes"])
        self.n_receptors = int(params["tau_syn"].shape[1])

        node_type_ids_np = network["node_type_ids"].astype(np.int64)
        self.register_buffer(
            "node_type_ids",
            torch.as_tensor(node_type_ids_np, dtype=torch.long, device=self.device),
        )

        voltage_scale_np = params["V_th"] - params["E_L"]
        voltage_offset_np = params["E_L"]

        def gather_np(a):
            return torch.as_tensor(a[node_type_ids_np], dtype=torch.float32, device=self.device)

        voltage_scale = gather_np(voltage_scale_np)
        voltage_offset = gather_np(voltage_offset_np)
        self.register_buffer("voltage_scale", voltage_scale)
        self.register_buffer("voltage_offset", voltage_offset)

        self.register_buffer("v_th", (gather_np(params["V_th"]) - voltage_offset) / voltage_scale)
        self.register_buffer("e_l", (gather_np(params["E_L"]) - voltage_offset) / voltage_scale)
        self.register_buffer("v_reset", (gather_np(params["V_reset"]) - voltage_offset) / voltage_scale)
        self.register_buffer("param_g", gather_np(params["g"]))
        self.register_buffer("t_ref", gather_np(params["t_ref"]))

        asc_amps = gather_np(params["asc_amps"]) / voltage_scale[:, None]
        self.register_buffer("asc_amps", asc_amps)

        k = gather_np(params["k"])
        self.register_buffer("param_k", k)

        tau = gather_np(params["C_m"]) / gather_np(params["g"])
        decay = torch.exp(torch.tensor(-self.dt, device=self.device) / tau)
        current_factor = (1.0 / gather_np(params["C_m"])) * (1.0 - decay) * tau
        self.register_buffer("decay", decay)
        self.register_buffer("current_factor", current_factor)

        tau_syn = gather_np(params["tau_syn"])
        self.register_buffer("syn_decay", torch.exp(torch.tensor(-self.dt, device=self.device) / tau_syn))
        self.register_buffer("psc_initial", torch.e / tau_syn)

        # Register as buffers so they follow module.to(device) under Lightning/DDP.
        self.register_buffer(
            "gauss_std",
            torch.tensor(float(gauss_std), dtype=torch.float32, device=self.device),
        )
        self.register_buffer(
            "dampening_factor",
            torch.tensor(float(dampening_factor), dtype=torch.float32, device=self.device),
        )

        delays_np = network["synapses"]["delays"]
        self.max_delay = int(np.round(np.min([np.max(delays_np), max_delay])))

        rec_indices = network["synapses"]["indices"].copy()
        rec_weights = network["synapses"]["weights"].astype(np.float32).copy()
        rec_delays = network["synapses"]["delays"].copy()

        rec_target_type = node_type_ids_np[rec_indices[:, 0] // self.n_receptors]
        rec_weights = rec_weights / voltage_scale_np[rec_target_type]

        delay_steps = np.round(np.clip(rec_delays, self.dt, self.max_delay) / self.dt).astype(np.int64)
        rec_indices[:, 1] = rec_indices[:, 1] + self.n_neurons * (delay_steps - 1)

        rec_shape = (self.n_receptors * self.n_neurons, self.max_delay * self.n_neurons)
        rec_sparse = make_sparse(rec_indices, rec_weights * recurrent_weight_scale, rec_shape, self.device)
        self.register_buffer("rec_indices", rec_sparse.indices())
        self.rec_shape = rec_shape

        rec_values = rec_sparse.values()
        self.register_buffer("rec_sign", torch.sign(rec_values))
        self.recurrent_weight_values = nn.Parameter(rec_values.clone(), requires_grad=train_recurrent)

        in_indices = input_population["indices"].copy()
        in_weights = input_population["weights"].astype(np.float32).copy()
        in_target_type = node_type_ids_np[in_indices[:, 0] // self.n_receptors]
        in_weights = in_weights / voltage_scale_np[in_target_type]

        in_shape = (self.n_receptors * self.n_neurons, input_population["n_inputs"])
        in_sparse = make_sparse(in_indices, in_weights * input_weight_scale, in_shape, self.device)
        self.register_buffer("in_indices", in_sparse.indices())
        self.in_shape = in_shape

        in_values = in_sparse.values()
        self.register_buffer("in_sign", torch.sign(in_values))
        self.input_weight_values = nn.Parameter(in_values.clone(), requires_grad=train_input)

        bkg = np.asarray(bkg_weights, dtype=np.float32)
        bkg = bkg / np.repeat(voltage_scale_np[node_type_ids_np], self.n_receptors)
        self.bkg_weights = nn.Parameter(
            torch.as_tensor(bkg * 10.0, dtype=torch.float32, device=self.device),
            requires_grad=train_bkg,
        )

    def constrained_values(self, values, signs):
        if not self.use_dale_law:
            return values
        return torch.where(signs >= 0, F.relu(values), -F.relu(-values))

    def _runtime_device(self):
        # Use buffer device at runtime; `self.device` can be stale after .to(...)
        return self.v_reset.device

    def recurrent_sparse(self):
        vals = self.constrained_values(self.recurrent_weight_values, self.rec_sign)
        return torch.sparse_coo_tensor(
            self.rec_indices, vals, self.rec_shape, device=self._runtime_device(), check_invariants=False
        ).coalesce()

    def input_sparse(self):
        vals = self.constrained_values(self.input_weight_values, self.in_sign)
        return torch.sparse_coo_tensor(
            self.in_indices, vals, self.in_shape, device=self._runtime_device(), check_invariants=False
        ).coalesce()

    def zero_state(self, batch_size):
        B, N, R, D = batch_size, self.n_neurons, self.n_receptors, self.max_delay
        dev = self._runtime_device()
        return (
            torch.zeros(B, N * D, device=dev),
            self.v_reset[None, :].repeat(B, 1),
            torch.zeros(B, N, device=dev),
            torch.zeros(B, N, device=dev),
            torch.zeros(B, N, device=dev),
            torch.zeros(B, N * R, device=dev),
            torch.zeros(B, N * R, device=dev),
        )

    def project_input(self, x_t):
        return sparse_mm_batch(self.input_sparse(), x_t)

    def step_from_current(self, inputs, state):
        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state
        B = inputs.shape[0]
        N, R, D = self.n_neurons, self.n_receptors, self.max_delay

        shaped_z_buf = z_buf.reshape(B, D, N)
        prev_z = shaped_z_buf[:, 0]

        psc_rise = psc_rise.reshape(B, N, R)
        psc = psc.reshape(B, N, R)

        i_rec = sparse_mm_batch(self.recurrent_sparse(), z_buf)
        rec_inputs = (i_rec + inputs).reshape(B, N, R)

        new_psc_rise = self.syn_decay[None, :, :] * psc_rise + rec_inputs * self.psc_initial[None, :, :]
        new_psc = psc * self.syn_decay[None, :, :] + self.dt * self.syn_decay[None, :, :] * psc_rise

        new_r = F.relu(r + prev_z * self.t_ref[None, :] - self.dt)

        new_asc_1 = torch.exp(-self.dt * self.param_k[:, 0])[None, :] * asc_1 + prev_z * self.asc_amps[:, 0][None, :]
        new_asc_2 = torch.exp(-self.dt * self.param_k[:, 1])[None, :] * asc_2 + prev_z * self.asc_amps[:, 1][None, :]

        reset_current = prev_z * (self.v_reset[None, :] - self.v_th[None, :])
        input_current = psc.sum(dim=-1)
        decayed_v = self.decay[None, :] * v

        gathered_g = self.param_g * self.e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g[None, :]
        new_v = decayed_v + self.current_factor[None, :] * c1 + reset_current

        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th[None, :]) / normalizer[None, :]

        new_z = spike_gauss(v_sc, self.gauss_std, self.dampening_factor)
        new_z = torch.where(new_r > 0.0, torch.zeros_like(new_z), new_z)

        new_z_buf = torch.cat((new_z[:, None, :], shaped_z_buf[:, :-1, :]), dim=1).reshape(B, N * D)

        new_state = (
            new_z_buf,
            new_v,
            new_r,
            new_asc_1,
            new_asc_2,
            new_psc_rise.reshape(B, N * R),
            new_psc.reshape(B, N * R),
        )
        v_out = new_v * self.voltage_scale[None, :] + self.voltage_offset[None, :]
        return (new_z, v_out), new_state

    def forward(self, x, state=None, input_is_current=False):
        # x: [B, T, n_lgn] if input_is_current=False
        # x: [B, T, 4*n_neurons] if input_is_current=True
        B, T = x.shape[:2]
        if state is None:
            state = self.zero_state(B)

        zs, vs = [], []
        for t in range(T):
            if input_is_current:
                current_t = x[:, t]
            else:
                current_t = self.project_input(x[:, t])

            # same random background style as original SparseLayer when use_decoded_noise=False
            rest = (torch.rand(B, 10, device=x.device) < 0.1).float().sum(dim=-1)
            noise = self.bkg_weights[None, :] * rest[:, None] / 10.0
            current_t = current_t + noise

            (z, v), state = self.step_from_current(current_t, state)
            zs.append(z)
            vs.append(v)

        return torch.stack(zs, dim=1), torch.stack(vs, dim=1), state
