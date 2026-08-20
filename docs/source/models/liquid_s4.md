# Liquid-S4

Liquid-S4 {cite}`hasani2023liquids4` starts from a completely different
place than LTC/CfC: a **structured state-space model** (S4-style), which is
discrete-time and normally has a *fixed* linear transition. Liquid-S4's
contribution is making that transition input-dependent -- borrowing the
"liquid" idea, but grafting it onto a linear recurrence instead of an ODE.

## The equation

A plain diagonal S4/S4D-style layer runs the linear recurrence

$$
x_k = A\,x_{k-1} + B\,u_k, \qquad y_k = C\,x_k
$$

with a **fixed** diagonal $A$. `models/liquid_s4/model.py` implements the
liquid version of this: $A$ is replaced by a per-step effective decay that's
gated by the current input,

$$
a_{\text{eff}}(u_k) = e^{a_{\text{base}}} \odot \sigma\bigl(W_{\text{liquid}}\,u_k + b_{\text{liquid}}\bigr), \qquad
x_k = a_{\text{eff}}(u_k) \odot x_{k-1} + B\,u_k
$$

$a_{\text{base}} = -e^{\text{log\_decay}}$ is a per-unit, HiPPO-inspired base
decay rate spread on a log scale (so different state units specialize to
different timescales by default, before any input-dependent modulation), and
$\sigma(\cdot) \in (0, 1)$ is the "liquid" correction: a strong input can
temporarily slow a unit's decay down towards its base rate, or (since
$\sigma < 1$ always shrinks $a_{\text{eff}}$ below $e^{a_{\text{base}}}$)
sharpen the forgetting of state units that aren't currently relevant.

## How it's built

`LiquidS4Layer.forward` in
[`models/liquid_s4/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/liquid_s4/model.py):

```python
log_decay = torch.linspace(math.log(0.5), math.log(0.001), state_size)  # HiPPO-ish spread
a_base = -torch.exp(self.log_decay)                                      # always < 0 => stable

for t in range(seq_len):
    liquid_mod = torch.sigmoid(self.liquid_gate(u_t))    # input-dependent correction, in (0,1)
    a_eff = torch.exp(a_base) * liquid_mod                # the actual per-step decay used
    x = a_eff * x + self.B(u_t)
    ys.append(self.C(x) + self.D * u_t)
```

Unlike {doc}`ltc`/{doc}`cfc`, there's no `dt`/elapsed-time argument anywhere
here -- one call is one discrete token, full stop (see
{doc}`../model_comparison`). This is also a **sequential Python loop**, not
the paper's FFT convolution: the official Liquid-S4 trains by rewriting the
whole-sequence linear recurrence as one big convolution, computed in the
frequency domain, which is what lets a linear SSM train as fast as it does.
That rewrite depends on $A$ being *fixed* per layer, though -- Liquid-S4's
whole point is that $A$ *isn't* fixed here, so the official implementation
uses a more involved (still fast) scheme. This reimplementation keeps the
easy-to-read sequential loop and trades the FFT-convolution speed for
clarity; see {doc}`lrcssm` for a model in this repo that *does* ship a fast
parallel path.

## Multi-timescale memory, visualized

The base decay rates `a_base` are spread on a log scale specifically so
different state units remember over different horizons. The plot below
shows the (input-independent) impulse response $e^{a_{\text{base}} \cdot k}$
for a few of those units -- i.e., how much a unit that received one pulse at
$k=0$ still "remembers" $k$ steps later, absent any liquid modulation:

```{eval-rst}
.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    state_size = 32
    log_decay = np.linspace(np.log(0.5), np.log(0.001), state_size)
    a_base = -np.exp(log_decay)

    k = np.arange(0, 60)
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx in [0, 8, 16, 24, 31]:
        response = np.exp(a_base[idx] * k)
        ax.plot(k, response, label=f"unit {idx}  (a_base={a_base[idx]:.3f})")

    ax.set_xlabel("steps since input pulse  k")
    ax.set_ylabel("impulse response  exp(a_base * k)")
    ax.set_title("Liquid-S4: multi-timescale base decay (before liquid gating)")
    ax.legend(fontsize=8)
```

Units near index 0 (small $|a_{\text{base}}|$) decay almost instantly --
short-term memory; units near index 31 decay over dozens of steps --
long-term memory. The liquid gate then modulates *around* this fixed
spread on a per-input basis.

## Try it

```bash
python models/liquid_s4/example.py --device auto     # ETTh1 forecasting
```

or open [`models/liquid_s4/example.ipynb`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/liquid_s4/example.ipynb).
Full runnable code: [`models/liquid_s4/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/liquid_s4/model.py) ·
[`models/liquid_s4/README.md`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/liquid_s4/README.md).

## References

```{eval-rst}
.. bibliography::
   :filter: docname in docnames
```
