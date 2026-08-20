# How each model treats time

All five models are "liquid" in the same broad sense -- something in the
recurrence that would normally be fixed (a time constant, a decay rate, a
state-transition matrix) is instead computed *from the current input*. Where
they differ sharply is **how time itself enters the update**. That
difference is the main thing to understand before reading any one model's
page in detail.

| Model | Time is... | Update per step | Solver |
|---|---|---|---|
| [LTC](models/ltc) | continuous, numerically integrated | `ode_unfolds` (default 6) semi-implicit Euler sub-steps | fixed-point ODE solver, unrolled |
| [NCP](models/ncp) | continuous (same as LTC) | identical to LTC, restricted to sparse wiring | same fused ODE solver |
| [CfC](models/cfc) | continuous, but solved *in closed form* | one direct evaluation of an analytic gate | none -- no solver at all |
| [Liquid-S4](models/liquid_s4) | discrete (one token = one step) | linear decay `a_eff` modulated by the input | none -- plain linear recursion |
| [LrcSSM](models/lrcssm) | discrete (one token = one step) | non-linear-in-input, but *linear and diagonal in the state* | none in the loop -- but diagonal-in-state means the *whole sequence* can be solved by a parallel scan instead of a loop |

Two axes are doing all the work in that table:

1. **Continuous vs. discrete.** LTC/NCP/CfC descend from an ODE
   $\dot{x}(t) = \dots$ and treat consecutive inputs as samples of a
   continuous trajectory (elapsed time `dt` is a real input to the cell).
   Liquid-S4/LrcSSM treat the sequence as already discrete -- there's no
   `dt` in their update, just "the next token."
2. **Solved vs. closed-form vs. parallel-friendly.** LTC pays for being
   "genuinely" continuous by needing an iterative solver every step (`for
   _ in range(ode_unfolds)` in {doc}`models/ltc`). CfC removes the solver by
   using a closed-form approximation of the same ODE family. LrcSSM takes a
   different way out: instead of approximating, it *constrains* the update
   (diagonal Jacobian) so the exact recursion can be solved for an entire
   sequence at once with $O(\log T)$ sequential depth instead of $O(T)$ --
   see {doc}`models/lrcssm` for the scan itself.

## The liquid gate, side by side

The plot below reproduces each model's core scalar nonlinearity in isolation
(one hidden unit, everything else held fixed) so you can see the shape of
the "liquidity" each one actually learns. LTC and LrcSSM both gate on the
*input drive* (`w·x + b`, horizontal axis, left plot); CfC instead gates on
*elapsed time* for a few different learned rates (right plot), which is the
concrete difference between "integrate an ODE" and "close the form."

```{eval-rst}
.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    drive = np.linspace(-6, 6, 300)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -- left: LTC-style effective time constant, and LrcSSM-style decay,
    #    both as a function of input drive w*x + b.
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    f = sigmoid(drive)  # LTC/NCP synaptic activation
    for tau in [0.5, 1.0, 2.0]:
        tau_eff = 1.0 / (1.0 / tau + f)
        axes[0].plot(drive, tau_eff, label=f"LTC, tau={tau}")

    a_lrc = 1.0 - sigmoid(drive)  # LrcSSM decay a_k = 1 - g(u_k)
    axes[0].plot(drive, a_lrc, "k--", label="LrcSSM decay a_k")

    axes[0].set_xlabel("input drive  w*x + b")
    axes[0].set_ylabel("effective time constant / decay")
    axes[0].set_title("LTC & LrcSSM: gate on input")
    axes[0].legend(fontsize=8)

    # -- right: CfC-style gate sigmoid(-f*t) as a function of elapsed time,
    #    for a few different learned rates f.
    t = np.linspace(0, 5, 300)
    for f_val in [0.5, 1.5, 4.0]:
        gate = sigmoid(-f_val * t)
        axes[1].plot(t, gate, label=f"CfC, f(x)={f_val}")

    axes[1].set_xlabel("elapsed time  t")
    axes[1].set_ylabel("gate  sigmoid(-f(x)*t)")
    axes[1].set_title("CfC: closed-form gate on time")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
```

Reading the left plot: a small `f` (weak input drive) leaves the LTC
effective time constant close to the free-running `tau`, so the neuron
integrates slowly and "remembers." A strong drive pushes `f` up, `tau_eff`
down towards `1/f`, and the neuron reacts almost immediately to the current
input -- this is literally what makes the network's time constant "liquid."
LrcSSM's decay follows the same sigmoidal shape but is used directly as the
recursion's multiplicative decay rather than folded into a `1/(1/tau + f)`
expression.

Reading the right plot: CfC never integrates anything -- for a given learned
rate `f(x)`, the gate at elapsed time `t` is read off the curve directly.
Larger `f(x)` (steeper curve) means the cell locks onto the new candidate
state faster; this is the same qualitative behavior as a small LTC time
constant, produced without a solver.

## Benchmark it yourself

{doc}`benchmarks` runs all five on the same dataset and reports wall-clock
train time alongside accuracy/MSE -- a good next step after reading the
plots above is to actually measure how "closed-form" vs. "ODE-solved" vs.
"parallel-scan" plays out in practice on your machine:

```bash
python benchmarks/run_all.py --config benchmarks/configs/classification_suite.yaml --device auto
```
