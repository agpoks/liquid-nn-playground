# Neural ODE

**Paper:** Chen, Rubanova, Bettencourt, Duvenaud, *"Neural Ordinary
Differential Equations"*, NeurIPS 2018 --
[arXiv:1806.07366](https://arxiv.org/abs/1806.07366). See
[`papers/README.md`](../../papers/README.md) for how this baseline fits
alongside the liquid-network lineage.

## Idea in one paragraph

`dh/dt = f_theta(h, x)`, with `f_theta` an arbitrary small MLP -- no leak
term, no fixed structure, no built-in stability guarantee.
[`models/ltc`](../ltc)'s ODE *is* a Neural ODE: it's exactly this equation
with a specific, provably-bounded right-hand side substituted for `f_theta`.
This model is the unconstrained general case LTC specializes, and shows what
you give up (guaranteed stability, a closed-form-friendly structure) and
gain (no architectural assumptions at all) by not constraining the dynamics.
Integrated with a hand-written classic RK4 step rather than LTC's
semi-implicit Euler, since a fully generic `f_theta` has no linear structure
to exploit for an implicit solve; gradients flow by autodiffing directly
through the unrolled RK4 steps rather than the paper's memory-efficient
adjoint method (see `model.py` for why that's fine at this repo's sequence
lengths, and where the adjoint method actually matters).

## Files

- `model.py` -- `NeuralODECell` (hand-written 4th-order Runge-Kutta
  integrator over a learned MLP vector field) + `NeuralODEModel` (sequence
  wrapper for classification/regression).
- `example.py` -- trains on the UCI Ozone dataset (`--device {auto,cpu,cuda,mps}`),
  same task as [`models/ltc/example.py`](../ltc/example.py) for direct comparison.
- `example.ipynb` -- same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/node/example.py --device auto
# or open models/node/example.ipynb
```
