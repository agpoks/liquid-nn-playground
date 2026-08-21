# Vanilla RNN (Elman)

**Paper:** Elman, *"Finding Structure in Time"*, Cognitive Science, 1990.
A classic, pre-arXiv reference; see [`papers/README.md`](../../papers/README.md)
for how this baseline fits alongside the liquid-network lineage.

## Idea in one paragraph

`h_t = tanh(W_x x_t + W_h h_{t-1} + b)` -- the simplest possible recurrent
update. There is no notion of elapsed time at all: `h_t` depends only on the
*order* inputs arrive in, never on how much wall-clock time separates them.
This is the baseline every other model in this repo adds machinery on top
of -- see [`models/ctrnn`](../ctrnn) (adds a fixed time constant),
[`models/node`](../node) (adds a fully general learned ODE), and
[`models/ltc`](../ltc) (adds an *input-dependent* time constant, the point
where "liquid" starts).

## Files

- `model.py` -- `RNNCell` (one `tanh(Linear + Linear)` step) + `RNNModel`
  (sequence wrapper for classification/regression).
- `example.py` -- trains on the UCI Ozone dataset (`--device {auto,cpu,cuda,mps}`),
  same task as [`models/ltc/example.py`](../ltc/example.py) for direct comparison.
- `example.ipynb` -- same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/rnn/example.py --device auto
# or open models/rnn/example.ipynb
```
