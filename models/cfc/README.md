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

- `model.py` -- `CfCCell` (closed-form gated update) + `CfCModel`.
- `example.py` -- trains on UCI Person Activity (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/cfc/example.py --device auto
# or open models/cfc/example.ipynb
```
