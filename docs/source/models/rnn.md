# RNN -- the plain baseline

Before any notion of "liquid" or even "continuous time" enters the picture,
it's worth pinning down the simplest possible recurrent model: a vanilla
(Elman) RNN {cite}`elman1990findingstructure`. Every other model in this repo
adds exactly one piece of machinery on top of this baseline -- see
{doc}`../model_comparison` for the full ladder from here up to LTC.

## The equation

$$
h_t = \tanh\bigl(W_x x_t + W_h h_{t-1} + b\bigr)
$$

That's the whole update. There is no `dt`, no per-unit time constant, no
gate -- `h_t` depends only on the *order* inputs arrive in, never on how much
wall-clock time separates them. Compare this directly to {doc}`ltc`'s
$dx/dt = -x/\tau + f(x,I)(A-x)$: nothing here plays the role of $\tau$, $f$,
or $A$ at all.

## How it's built

`RNNCell.forward` in
[`models/rnn/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/rnn/model.py)
is two `Linear` layers (see {doc}`../getting_started` for what `Linear`
computes) and one `tanh`:

```python
def forward(self, x_t, h_prev, dt: float = 1.0):
    del dt  # accepted for interface parity with the other cells; unused
    return torch.tanh(self.w_in(x_t) + self.w_rec(h_prev))
```

```{eval-rst}
.. plot::

    from liquid_playground.utils.diagrams import new_ax, box, arrow, INPUT, LINEAR, NONLIN, STATE, OTHER

    fig, ax = new_ax(figsize=(8.0, 4.4), xlim=(0, 13), ylim=(0, 8))

    box(ax, 1.0, 6.0, 1.5, 1.0, "x_t", INPUT)
    box(ax, 1.0, 2.0, 1.7, 1.0, "h_prev", STATE)
    box(ax, 3.6, 6.0, 2.0, 1.0, "Linear\nw_in (+bias)", LINEAR)
    box(ax, 3.6, 2.0, 2.0, 1.0, "Linear\nw_rec (no bias)", LINEAR)
    box(ax, 6.2, 4.0, 1.4, 1.0, "+", OTHER)
    box(ax, 8.4, 4.0, 1.6, 1.0, "tanh", NONLIN)
    box(ax, 10.8, 4.0, 1.3, 0.9, "h_t", STATE)

    arrow(ax, (1.75, 6.0), (2.6, 6.0))
    arrow(ax, (1.85, 2.0), (2.6, 2.0))
    arrow(ax, (4.6, 6.0), (5.5, 4.35))
    arrow(ax, (4.6, 2.0), (5.5, 3.65))
    arrow(ax, (6.9, 4.0), (7.6, 4.0))
    arrow(ax, (9.2, 4.0), (10.15, 4.0))
    arrow(ax, (11.4, 4.55), (11.4, 3.95), curve=1.1, dashed=True)
    ax.text(12.35, 4.3, "t+1", fontsize=8, ha="center", color="#334155")
    arrow(ax, (11.35, 3.55), (12.5, 2.9))
    ax.text(12.55, 2.6, "readout\n(Linear) → y", fontsize=8, va="center", color="#334155")

    ax.set_title("Vanilla RNN cell: one input step", fontsize=11)
```

`RNNModel` loops this cell over the sequence exactly like every other model
here and reads out the final hidden state through one more `Linear`.

## Try it

```bash
python models/rnn/example.py --device auto     # same UCI Ozone task as models/ltc
```

or open [`models/rnn/example.ipynb`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/rnn/example.ipynb).
Full runnable code: [`models/rnn/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/rnn/model.py) ·
[`models/rnn/README.md`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/rnn/README.md).

## References

```{eval-rst}
.. bibliography::
   :filter: docname in docnames
```
