# CT-RNN (Continuous-Time RNN)

**Paper:** Funahashi & Nakamura, *"Approximation of dynamical systems by
continuous time recurrent neural networks"*, Neural Networks, 1993. A
classic, pre-arXiv reference; see [`papers/README.md`](../../papers/README.md)
for how this baseline fits alongside the liquid-network lineage.

## Idea in one paragraph

Each hidden unit obeys `dh/dt = (-h + tanh(W_x x + W_h h + b)) / tau`, with
`tau` a learnable per-unit time constant that is **fixed** -- it never
depends on the current input. [`models/ltc`](../ltc)'s
`dx/dt = -x/tau + f(x, I) * (A - x)` is exactly this equation's family with
the leak additionally gated by an input-dependent synapse `f`, so LTC's
*effective* time constant becomes `1 / (1/tau + f(x, I))` instead of the
constant `tau` used here. CT-RNN is what you get by freezing that gate --
there is no "liquid" property (no dependence of dynamics speed on the
current input) in this model at all, making it the natural "continuous time,
but not liquid" comparison point.

## Files

- `model.py` -- `CTRNNCell` (same fused semi-implicit Euler solver as
  [`models/ltc/model.py`](../ltc/model.py), applied to a fixed-tau leak) +
  `CTRNNModel` (sequence wrapper for classification/regression).
- `example.py` -- trains on the UCI Ozone dataset (`--device {auto,cpu,cuda,mps}`),
  same task as [`models/ltc/example.py`](../ltc/example.py) for direct comparison.
- `example.ipynb` -- same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/ctrnn/example.py --device auto
# or open models/ctrnn/example.ipynb
```
