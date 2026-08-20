# Liquid Time-Constant Networks (LTC)

**Paper:** Hasani, Lechner, Amini, Rus, Grosu, *"Liquid Time-constant Networks"*,
AAAI 2021 — [arXiv:2006.04439](https://arxiv.org/abs/2006.04439). Picked as the
clearest presentation of the core LTC ODE (Eq. 5-6) and its stability proof;
see [`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Every hidden unit is a leaky integrator whose time constant is *not* fixed:
`dx_i/dt = -x_i/tau_i + f_i(x, I) * (A_i - x_i)`, where `f_i` is a sigmoidal
synapse driven by the current input and recurrent state. Because `f_i` depends
on the input, the effective time constant `1 / (1/tau_i + f_i)` changes at
every step -- the network is "liquid": it dynamically speeds up or slows down
depending on what it's looking at, and this makes it robust to non-uniform /
irregularly sampled time series.

## Files

- `model.py` -- `LTCCell` (fused semi-implicit Euler ODE solver) + `LTCModel`
  (sequence wrapper for classification/regression).
- `example.py` -- trains on the UCI Ozone dataset (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/ltc/example.py --device auto
# or open models/ltc/example.ipynb
```
