# Closed-form Continuous-time Networks (CfC)

**Paper:** Hasani, Lechner, Amini, Ray, Chahine, Wang, Rus, *"Closed-form
Continuous-time Neural Networks"*, Nature Machine Intelligence 2022 —
[arXiv:2106.13898](https://arxiv.org/abs/2106.13898) /
[nature.com](https://www.nature.com/articles/s42256-022-00556-7) (open access).
Picked because it gives an explicit closed-form state update (no ODE solver)
and is the clearest place the "liquid gate" is written as one equation; see
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

CfC approximates the solution of the LTC ODE in closed form instead of
integrating it numerically: `h(t) = sigma(-f(x)*t) * g(x) + (1 - sigma(-f(x)*t)) * h(x)`.
`f`, `g`, `h` are small feed-forward heads over a shared backbone; `sigma(-f(x)*t)`
is the input-dependent gate that plays the role of LTC's varying time constant.
Because there's no unrolled ODE solver, CfC is reported as >100x faster to
train/run than LTC-style networks at comparable accuracy.

## Files

- `model.py` -- `CfCCell` (closed-form gated update) + `CfCModel`. **PyTorch.**
- `example.py` -- trains on UCI Person Activity (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough with loss/accuracy plots.
- `model_jax.py` / `example_jax.py` -- the identical architecture and task,
  ported to **JAX/Flax**, with JAX's own `--device {auto,cpu,gpu,tpu}`
  selection. CfC is the one model with a JAX port in this repo (see
  [the repo README](../../README.md) for why just this one) -- useful if you
  want to compare PyTorch vs. JAX training speed/ergonomics on the same model.

## Run it

```bash
pip install -e .
python models/cfc/example.py --device auto
# or open models/cfc/example.ipynb

# JAX version:
pip install -e ".[jax]"
python models/cfc/example_jax.py --device auto
```
