# Liquid-S4

**Paper:** Hasani, Lechner, Wang, Chahine, Amini, Rus, *"Liquid Structural
State-Space Models"*, ICLR 2023 — [arXiv:2209.12951](https://arxiv.org/abs/2209.12951).
Picked as the clearest write-up of making an S4-style diagonal state-space
model's transition matrix input-dependent; see
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Structured state-space models (S4) run a linear recurrence
`x_k = A x_{k-1} + B u_k` with a *fixed* transition matrix `A`, which is what
lets them be trained as one big FFT convolution. Liquid-S4 makes `A` respond
to the input like LTC/CfC do, closing some of the accuracy gap to fully liquid
(ODE-based) networks while keeping most of S4's efficiency. This
reimplementation applies that idea with a diagonal, HiPPO-inspired base decay
gated by a sigmoid of the current input at every step.

> This is a sequential-scan educational version, not the paper's FFT-parallel
> training path or complex-HiPPO init -- see
> [raminmh/liquid-s4](https://github.com/raminmh/liquid-s4) for the exact,
> LRA-benchmarked implementation.

## Files

- `model.py` -- `LiquidS4Layer` (diagonal SSM + liquid gate) + `LiquidS4Model`.
- `example.py` -- trains on ETTh1 forecasting (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough with a forecast plot.

## Run it

```bash
pip install -e .
python models/liquid_s4/example.py --device auto
# or open models/liquid_s4/example.ipynb
```
