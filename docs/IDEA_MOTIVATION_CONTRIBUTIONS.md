# Membrane-Gated Attention: Motivation and Contributions

This note records the current top-conference-style framing of the project idea.
It is intentionally written as paper material rather than implementation notes.

## One-Sentence Positioning

Membrane-Gated Attention (MGA) uses the pre-threshold membrane potential of
spiking neurons as a hardware-friendly control signal for temporal attention
memory, enabling recurrent spike-driven attention without dense floating-point
gates on the recurrence path.

Alternative phrasing:

> We show that the membrane potential, a latent state already computed in
> spiking neurons, can serve as an efficient control signal for temporal
> attention memory, enabling hardware-friendly recurrent attention without
> dense floating-point gates.

## Motivation

Spiking neural networks are attractive for event-driven perception because they
process sparse binary events over time and are naturally aligned with
neuromorphic hardware. Recent spiking transformers improve representation power
by bringing attention-style token mixing into SNNs.

However, temporal memory in spiking attention remains underdeveloped.
Memoryless spike-driven linear attention preserves sparse computation but
computes each timestep independently, discarding cross-time evidence. This can
be sufficient for short or saturated event-recognition tasks, but it is less
suitable for long-horizon spike streams where evidence must accumulate over
many timesteps.

A natural solution is to introduce a recurrent key-value attention state with a
content-dependent gate. This creates a temporal-memory dilemma:

- Memoryless spike-driven attention is hardware-friendly but temporally limited.
- Low-rank continuous gates restore temporal memory but reintroduce dense
  floating-point multiplications.
- Spike-only heuristic gates are cheap but operate after thresholding and
  discard subthreshold evidence.

The key observation behind MGA is that the membrane potential is not merely an
internal variable for spike generation. It is a temporally accumulated,
subthreshold representation that is already computed by LIF neurons. Unlike
binary spikes, the pre-threshold membrane preserves evidence before a spike is
emitted. This makes it a natural control signal for attention memory.

## Core Insight

The membrane potential offers a missing middle ground between dense continuous
gates and overly quantized spike-only gates. It provides analog temporal
information already present inside the spiking neuron dynamics, while the final
gate decisions can remain binary and event-driven.

In short:

> MGA does not add a separate dense gate network; it repurposes the neuron's own
> membrane dynamics as the attention memory controller.

## Method Framing

MGA maintains a recurrent key-value state:

```text
S_t = alpha_eff * S_{t-1} + w_h * s_beta * K_t^T V_t
alpha_eff = 1 - s_gamma * 2^{-k_bits}
```

where:

- `s_gamma` is produced by a LIF forgetting gate driven by pooled
  pre-threshold key membrane.
- `s_beta` is produced by a LIF writing gate driven by key spike-rate
  statistics.
- `w_h` is a learned per-head write scale, initialized to 0.125.
- The pooled membrane input is normalized per head before the gamma LIF.
- `2^{-k_bits}` enables shift-based decay, e.g. `S - (S >> k)` in fixed-point
  hardware.
- The shared Q/K/V/projection backbone remains identical across attention
  conditions.

The conceptual distinction from baselines is:

- C0 / SDLA: memoryless, hardware-friendly, no cross-time recurrent state.
- C1 / low-rank gate: adaptive recurrent memory, but continuous gate compute.
- C2 / one-minus-k gate: cheap recurrent gate, but uses spike-only information.
- C3 / MGA: adaptive recurrent memory controlled by pre-threshold membrane
  dynamics and binary LIF gates.

## Contribution Bullets

Suggested paper-ready contribution statement:

> In summary, this work makes the following contributions:

1. We identify the temporal-memory dilemma of spiking attention: memoryless
   spike-driven attention is hardware-friendly but temporally limited, while
   recurrent continuous gates improve memory at the cost of dense floating-point
   computation.

2. We propose Membrane-Gated Attention, a recurrent spike-driven attention
   mechanism that uses pre-threshold key membrane potentials and LIF gate
   neurons to control memory decay and state writing.

3. We derive a hardware-aware update rule in which recurrent state decay can be
   mapped to binary masking and bit-shift operations in fixed-point deployment.

4. We establish a controlled evaluation protocol against memoryless, low-rank,
   and spike-only gates, and use spatio-temporal effective receptive field
   diagnostics to measure temporal memory propagation in spiking attention.

5. Exploratory pilot results identify SHD T=100 as a promising setting for MGA,
   motivating confirmatory multi-seed experiments and mechanism ablations.

## Conservative Abstract Draft

Spiking neural networks promise event-driven and hardware-efficient
computation, yet recent spiking transformers still lack an efficient mechanism
for adaptive temporal memory. Memoryless spike-driven linear attention preserves
sparsity but discards cross-time evidence, whereas recurrent gated variants
typically rely on dense floating-point gates or spike-only heuristics that
either undermine neuromorphic efficiency or ignore subthreshold dynamics. We
propose Membrane-Gated Attention (MGA), a recurrent spike-driven attention
module that repurposes the pre-threshold membrane potential of key neurons as a
control signal for temporal memory. MGA employs two LIF-inspired binary gates:
a membrane-driven forgetting gate that decays the recurrent key-value state
through shift-based updates, and a spike-rate-driven writing gate that controls
state injection. This design preserves the shared spike-driven Q/K/V backbone
while enabling a shift/subtract mapping for recurrent state decay. We
further introduce a controlled evaluation protocol and a spatio-temporal
effective receptive field diagnostic to quantify temporal memory propagation.
Exploratory experiments on DVS128-Gesture and SHD motivate a focused evaluation
of long-horizon behavior; confirmatory multi-seed, ablation, and hardware results
remain required.

## How To Present Current Pilot Results

Use conservative claims:

- DVS128-Gesture is a short-horizon and near-saturated control task. MGA remains
  competitive but does not beat the strongest low-rank gate.
- SHD is more diagnostic of temporal memory. The current T=100 single-seed result
  is promising, but T=25/50 is non-monotonic and T=200 is incomplete.
- The main evidence is not only accuracy. The paper should jointly report
  accuracy, ST-ERF, gate parameters, and recurrence gate-path FP multiplications.

Avoid overclaiming:

- Do not claim comprehensive accuracy superiority across all tasks unless
  follow-up experiments support it.
- Do not claim the whole model has no floating-point multiplications. The
  precise claim is a multiply-free fixed-point mapping of recurrent state decay,
  excluding gate-LIF and shared Q/K/V/projection/MLP work.
- Do not use DVS128-Gesture as the main long-horizon evidence because its
  timestep horizon is short and the baseline is already saturated.

## Stronger Claim If Follow-Up Experiments Succeed

If SHD temporal-horizon sweeps, C3 ablations, and SeqMNIST/PSeqMNIST results
show a clearer advantage, the headline can become:

> MGA improves long-horizon temporal modeling while reducing recurrent
> gate-path floating-point multiplications compared with low-rank continuous
> gates.

Potential stronger contribution wording:

1. We reveal a fundamental mismatch between dense recurrent gates and the
   hardware-efficiency objective of spiking transformers.
2. We introduce MGA, which converts LIF membrane dynamics into a binary
   event-driven controller for attention memory.
3. We show that MGA improves long-horizon temporal modeling while reducing
   recurrence gate-path compute compared with low-rank continuous gates.
4. We validate MGA across event vision, audio spike streams, and long sequence
   benchmarks, demonstrating improved temporal ERF and competitive or superior
   accuracy under matched-capacity settings.

## Candidate Titles

1. Membrane-Gated Attention for Hardware-Friendly Spiking Transformers
2. Event-Driven Temporal Memory in Spiking Transformers via Membrane-Gated
   Attention
3. Subthreshold Membranes as Gates: Efficient Temporal Attention for Spiking
   Neural Networks
4. Towards Multiply-Free Recurrent Attention in Spiking Transformers
5. Membrane Dynamics Are Efficient Attention Gates for Spiking Sequence Modeling
