# CfC -- Closed-form Continuous-time Networks

CfC {cite}`hasani2022cfc` starts from the same ODE family as LTC (see
{doc}`ltc`) and asks: instead of unrolling a solver every step, can we write
down an approximate *closed-form solution* directly? The answer is yes, and
it's ~100x faster at both train and inference time for a small accuracy
cost. `models/cfc/model.py` implements exactly that closed form.

## The equation

The closed-form state update (paper Eq. 8-10 in {cite}`hasani2022cfc`) is:

$$
h(t) = \sigma\!\bigl(-f(x)\,t\bigr)\, g(x) \;+\; \Bigl(1 - \sigma\!\bigl(-f(x)\,t\bigr)\Bigr)\, h(x)
$$

where $f$, $g$, $h$ (confusingly reusing the letter $h$ for a *projection
head*, not the hidden state -- the code below calls it `h_proj` to avoid the
clash) are small feed-forward heads over a shared backbone, $\sigma$ is the
logistic sigmoid, and $t$ is the elapsed time since the last input. Compare
this to LTC's effective time constant
$\tau_{\text{eff}} = 1/(1/\tau + f)$: $\sigma(-f(x)t)$ plays the same
gating role -- large $f(x)$ collapses the gate to 0 quickly (fast forgetting
of the old state, same as a short $\tau_{\text{eff}}$), small $f(x)$ keeps
the gate near 1 (slow forgetting) -- but it's read directly off a formula
instead of being integrated.

## How it's built

`CfCCell.forward` in
[`models/cfc/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/cfc/model.py)
maps onto the equation term for term:

```python
z = self.backbone(torch.cat([x_t, h_prev], dim=-1))   # shared features of (input, state)
f = self.f_head(z)                                      # f(x) -- the rate
g = torch.tanh(self.g_head(z))                          # g(x) -- new-state candidate
h_proj = torch.tanh(self.h_head(z))                     # h(x) -- retained-state candidate

gate = torch.sigmoid(-f * (self.time_scale.abs() + 1e-3) * dt)
h_new = gate * h_proj + (1 - gate) * g
```

`self.time_scale` is a learnable per-unit rescaling of $t$, so different
hidden units can learn to react on different timescales even though they
all see the same wall-clock `dt` -- the same multi-timescale idea LTC gets
from per-unit $\tau_i$, but folded into the gate instead of a separate
parameter that needs integrating. Because this is one direct formula (no
loop over sub-steps like {doc}`ltc`), `CfCModel.forward` calls the cell
exactly once per input step. See {doc}`../model_comparison` for the
side-by-side gate-shape plot.

## Try it

```bash
python models/cfc/example.py --device auto           # PyTorch, trains on UCI Person Activity
python models/cfc/example_jax.py --device auto        # identical model in JAX/Flax
```

or open [`models/cfc/example.ipynb`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/cfc/example.ipynb).
Full runnable code: [`models/cfc/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/cfc/model.py) ·
[`models/cfc/model_jax.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/cfc/model_jax.py) ·
[`models/cfc/README.md`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/cfc/README.md).

## References

```{eval-rst}
.. bibliography::
   :filter: docname in docnames
```
