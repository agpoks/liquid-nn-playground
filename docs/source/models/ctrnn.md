# CT-RNN -- continuous time, but not liquid yet

CT-RNN {cite}`funahashi1993ctrnn` is the direct ancestor {doc}`ltc` generalizes:
it's a genuine ODE, integrated the same way LTC is, but its time constant is
**fixed** -- it never depends on the current input. This is the model that
isolates exactly one variable in {doc}`../model_comparison`'s ladder:
"continuous time" without "liquid."

## The equation

$$
\frac{dh}{dt} = \frac{-h + \tanh\bigl(W_x x + W_h h + b\bigr)}{\tau}
$$

with $\tau$ a learnable **per-unit but input-independent** time constant.
Compare directly to {doc}`ltc`'s
$dx/dt = -x/\tau + f(x, I)(A - x)$: rearranged, LTC's *effective* time
constant is $1/(1/\tau + f(x,I))$ -- an input-dependent correction on top of
exactly this equation's fixed $\tau$. Freeze $f$ at a constant and fold $A$
into the $\tanh$ nonlinearity, and LTC's ODE collapses to this one. CT-RNN is
what a liquid network looks like with the "liquid" part switched off.

## How it's built

`CTRNNCell.forward` in
[`models/ctrnn/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ctrnn/model.py)
reuses {doc}`ltc`'s exact fused semi-implicit (backward) Euler scheme --
same `ode_unfolds`, same unconditional stability -- just solving a different
(fixed-$\tau$) right-hand side:

```python
tau = softplus(self.tau) + 1e-3
sub_dt = dt / self.ode_unfolds

h = h_prev
for _ in range(self.ode_unfolds):
    target = tanh(self.w_in(x_t) + self.w_rec(h))   # frozen for this sub-step
    h = (h + (sub_dt / tau) * target) / (1.0 + sub_dt / tau)
```

That is the backward-Euler solution of $\dot h = (-h + \text{target})/\tau$
for $h$ given the previous value -- line-for-line the same shape as
{doc}`ltc`'s `numerator / denominator` update, just without an $f$ or $A$
anywhere in it:

```{eval-rst}
.. plot::

    from liquid_playground.utils.diagrams import new_ax, box, arrow, INPUT, LINEAR, NONLIN, STATE, OTHER

    fig, ax = new_ax(figsize=(9.0, 4.8), xlim=(0, 14.5), ylim=(0, 9))

    box(ax, 1.0, 6.5, 1.5, 1.0, "x_t", INPUT)
    box(ax, 1.0, 2.5, 1.7, 1.0, "h_prev", STATE)
    box(ax, 3.6, 6.5, 2.0, 1.0, "Linear\nw_in (+bias)", LINEAR)
    box(ax, 3.6, 2.5, 2.0, 1.0, "Linear\nw_rec (no bias)", LINEAR)
    box(ax, 6.4, 4.5, 1.4, 1.0, "+", OTHER)
    box(ax, 8.6, 4.5, 1.9, 1.0, "tanh\n= target", NONLIN)
    box(ax, 11.6, 4.5, 2.3, 2.2, "semi-implicit\nEuler, fixed τ\n(x6 sub-steps)", STATE)
    box(ax, 11.6, 7.9, 1.4, 0.9, "h_t", STATE)

    arrow(ax, (1.75, 6.5), (2.6, 6.5))
    arrow(ax, (1.85, 2.5), (2.6, 2.5))
    arrow(ax, (4.6, 6.5), (5.75, 4.85))
    arrow(ax, (4.6, 2.5), (5.75, 4.15))
    arrow(ax, (7.1, 4.5), (7.65, 4.5))
    arrow(ax, (9.55, 4.5), (10.45, 4.5))
    ax.text(10.0, 4.85, "τ", fontsize=8, ha="center", color="#334155")
    arrow(ax, (11.6, 5.6), (11.6, 7.45))
    arrow(ax, (12.3, 8.3), (12.3, 7.6), curve=1.1, dashed=True)
    ax.text(13.35, 7.95, "t+1", fontsize=8, ha="center", color="#334155")
    arrow(ax, (12.3, 7.55), (13.5, 7.55))
    ax.text(13.65, 7.55, "readout\n(Linear) → y", fontsize=8, va="center", color="#334155")

    ax.set_title("CT-RNN cell: one input step", fontsize=11)
```

## Try it

```bash
python models/ctrnn/example.py --device auto     # same UCI Ozone task as models/ltc
```

or open [`models/ctrnn/example.ipynb`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ctrnn/example.ipynb).
Full runnable code: [`models/ctrnn/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ctrnn/model.py) ·
[`models/ctrnn/README.md`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ctrnn/README.md).

## References

```{eval-rst}
.. bibliography::
   :filter: docname in docnames
```
