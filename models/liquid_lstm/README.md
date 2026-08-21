# Liquid-LSTM

**Not a paper reimplementation** -- unlike every other model in `models/`,
there is no single "Liquid-LSTM" paper. This is a hybrid built for this repo,
combining:

- Hochreiter & Schmidhuber, *"Long Short-Term Memory"*, Neural Computation,
  1997 -- the four-gate (forget/input/candidate/output) LSTM cell.
- Hasani, Lechner, Amini, Rus, Grosu, *"Liquid Time-constant Networks"*,
  AAAI 2021 -- [`models/ltc`](../ltc)'s continuous-time, input-gated leak.

See [`papers/README.md`](../../papers/README.md) for both references and how
this hybrid fits alongside the paper-based models and baselines.

## Idea in one paragraph

Standard LSTM computes four gates and applies one discrete cell-state
update `c_t = f_t * c_prev + i_t * g_t`. Liquid-LSTM keeps all four gates
(so it still has LSTM's separate write/erase/read control, unlike LTC's
single synapse) but replaces that discrete update with LTC's continuous-time
ODE, using the forget gate to build an input-dependent effective time
constant the same way LTC's synapse does:
`dc/dt = -c/tau_eff + i_t * g_t`, with
`1/tau_eff = 1/tau + (1 - f_t)`. `f_t` near 1 ("remember") keeps `tau_eff`
close to the free-running `tau` (slow decay); `f_t` near 0 ("forget") pushes
`tau_eff` down (fast decay) -- the same qualitative behavior as a standard
forget gate, now realized as a genuinely continuous-time leak with its own
`dt`/`ode_unfolds` instead of one discrete multiply.

## Files

- `model.py` -- `LiquidLSTMCell` (four LSTM gates + LTC-style semi-implicit
  Euler cell-state solve) + `LiquidLSTMModel` (sequence wrapper).
- `example.py` -- trains on the UCI Ozone dataset (`--device {auto,cpu,cuda,mps}`),
  same task as [`models/ltc/example.py`](../ltc/example.py) for direct comparison.
- `example.ipynb` -- same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/liquid_lstm/example.py --device auto
# or open models/liquid_lstm/example.ipynb
```
