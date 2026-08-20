# LrcSSM (Liquid-Resistance Liquid-Capacitance State-Space Model)

**Paper:** Farsang, Hasani, Rus, Grosu, *"Parallelization of Non-linear
State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks
for Efficient Sequence Modeling"*, NeurIPS 2025 —
[arXiv:2505.21717](https://arxiv.org/abs/2505.21717). The newest model in
this repo. Picked as the clearest derivation of the diagonal-Jacobian
parallel-scan trick; see [`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Non-linear, input-dependent recurrences (LTC, CfC, ...) are normally forced
into a sequential `O(T)` loop because each step's Jacobian mixes all state
dimensions. LrcSSM constrains the state update to an RC-circuit form where the
Jacobian is *diagonal by construction* -- each state unit's next value depends
on its *own* previous value only, gated by input-dependent "resistance"/
"capacitance" terms. A diagonal linear recursion can be solved for an entire
sequence in parallel (`O(log T)` depth) with a cumulative-product/cumsum scan,
so LrcSSM keeps LTC-style non-linear, input-dependent dynamics while training
as fast as a linear SSM (S4/S5/Mamba-class). `model.py` implements both the
sequential loop and the parallel-scan solve and the example checks they agree.

> Educational reimplementation of the core mechanism, not the paper's fused
> CUDA parallel-scan kernel -- see
> [MoniFarsang/LrcSSM](https://github.com/MoniFarsang/LrcSSM) for that.

## Files

- `model.py` -- `LrcSSMLayer` (diagonal RC-circuit update, sequential +
  parallel-scan) + `LrcSSMModel`.
- `example.py` -- trains on ETTh1 forecasting, prints the sequential-vs-parallel
  agreement check (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough plus a forecast plot and a scan-speed
  comparison.

## Run it

```bash
pip install -e .
python models/lrcssm/example.py --device auto
# or open models/lrcssm/example.ipynb
```
