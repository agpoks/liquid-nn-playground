# CT-GRU (Continuous-Time Gated Recurrent Unit)

**Paper:** Mozer, Kazakov, Lindsey, *"Discrete Event, Continuous Time RNNs"*,
2017 -- [arXiv:1710.04110](https://arxiv.org/abs/1710.04110). Completes the
family the other baselines sketch: RNN (no time axis) -> CT-RNN (one fixed
time constant) -> **CT-GRU (a bank of fixed time constants, gated)** ->
Neural ODE / Liquid-LSTM -> [`models/ltc`](../ltc)/[`models/cfc`](../cfc)
(input-dependent time constants). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Where CT-RNN adds one learnable-but-fixed `tau`, CT-GRU instead keeps
`num_scales` *pre-specified*, log-spaced time constants `tau_tilde_1 < ... <
tau_tilde_M` (hyperparameters, never learned) and, per hidden unit, learns a
soft distribution over which of those scales to read from (`r`) and write to
(`s`) at every step -- so each unit's memory is really `M` parallel traces,
one per fixed scale, each decaying by the *exact* closed form
`exp(-dt / tau_tilde_i)` rather than an approximated/unrolled ODE solve like
[`models/ltc`](../ltc) or [`models/ctrnn`](../ctrnn) use.

## Files

- `model.py` -- `CTGRUCell` (multi-timescale memory traces + softmax
  scale-selection gates) + `CTGRUModel` (sequence wrapper).
- `example.py` -- trains on the UCI Ozone dataset (`--device {auto,cpu,cuda,mps}`),
  same task as [`models/ltc/example.py`](../ltc/example.py) for direct comparison.
- `example.ipynb` -- same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/ctgru/example.py --device auto
# or open models/ctgru/example.ipynb
```
