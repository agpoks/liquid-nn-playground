# LrcSSM -- Liquid-Resistance Liquid-Capacitance State-Space Model

LrcSSM {cite}`farsang2025lrcssm` is the newest model in this repo (NeurIPS
2025) and answers a specific complaint about {doc}`ltc`-style models: their
non-linear, input-dependent dynamics normally force a sequential `O(T)` loop,
because each step's Jacobian mixes every state dimension together. LrcSSM
keeps the non-linear, input-dependent character but *constrains* the update
so the Jacobian is diagonal by construction -- which turns out to be enough
to solve an entire sequence in parallel.

## The equation

Each state unit behaves like an RC (resistor-capacitor) circuit whose
resistance/capacitance are themselves gated by the input (paper Eq. 3-6 in
{cite}`farsang2025lrcssm`):

$$
a_k = 1 - g(u_k), \qquad b_k = g(u_k)\odot c(u_k), \qquad x_k = a_k \odot x_{k-1} + b_k
$$

with $g(u_k) = \sigma(W_g u_k + b_g) \in (0,1)$ acting as a "conductance"
and $c(u_k) = \tanh(W_c u_k + b_c)$ a candidate level. Because $a_k$
multiplies $x_{k-1}$ **elementwise** -- never mixing state indices -- the
Jacobian $\partial x_k/\partial x_{k-1} = \mathrm{diag}(a_k)$ is diagonal by
construction, for *any* input, unlike Liquid-S4's decay (which is also
diagonal, but only because the layer was designed that way, not because
it's forced to stay stable and diagonal under the paper's formal
guarantee -- see {cite}`farsang2025lrcssm` Sec. 4 for the gradient-stability
proof this buys).

## Why diagonal means parallel

A diagonal linear recursion $x_k = a_k x_{k-1} + b_k$ can be solved for an
entire sequence with an **associative (parallel) scan**: define the affine
composition operator $(A_1, B_1) \cdot (A_2, B_2) = (A_2 A_1,\, A_2 B_1 +
B_2)$, run an inclusive scan over $(a_k, b_k)$ pairs with this operator, and
position $k$ ends up holding exactly $x_k$ (since $x_{-1}=0$). That scan
has $O(\log T)$ sequential depth instead of $O(T)$. `_parallel_diagonal_scan`
in
[`models/lrcssm/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/lrcssm/model.py)
implements it as a Hillis-Steele doubling scan:

```python
offset = 1
while offset < seq_len:
    a_prev = cat([ones(offset), a[:-offset]])   # shift-by-offset, pad with identity
    b_prev = cat([zeros(offset), b[:-offset]])
    b = a * b_prev + b        # combine: (a, b) . (a_prev, b_prev)
    a = a * a_prev
    offset *= 2
return b   # b_k = x_k after the scan
```

`LrcSSMLayer.forward` exposes both this `parallel=True` path and a plain
`parallel=False` sequential loop computing the *identical* recursion one
step at a time -- `models/lrcssm/example.py` checks the two agree to
float32 precision before training, which is worth running once yourself:

```bash
python models/lrcssm/example.py --device auto
# prints: "sequential vs. parallel-scan max abs diff: 1.19e-07"
```

```{eval-rst}
.. plot::

    from liquid_playground.utils.diagrams import new_ax, box, arrow, INPUT, LINEAR, NONLIN, STATE, OTHER

    fig, ax = new_ax(figsize=(11.0, 5.2), xlim=(0, 19), ylim=(0, 9.8))

    box(ax, 1.0, 6.0, 1.6, 1.0, "u\n(B,T,D)", INPUT)
    box(ax, 3.7, 8.2, 1.9, 1.0, "Linear\ng_head", LINEAR)
    box(ax, 6.3, 8.2, 2.1, 1.0, "sigmoid → g\n(conductance)", NONLIN)
    box(ax, 3.7, 3.8, 1.9, 1.0, "Linear\nc_head", LINEAR)
    box(ax, 6.3, 3.8, 2.1, 1.0, "tanh → c\n(candidate)", NONLIN)
    box(ax, 9.2, 6.0, 1.9, 1.5, "a = 1-g\nb = g·c", OTHER)
    box(ax, 12.2, 6.0, 2.7, 2.1, "parallel scan\n(Hillis-Steele)\nO(log T) depth", STATE)
    box(ax, 15.4, 6.0, 1.3, 0.9, "x", STATE)
    box(ax, 15.4, 8.7, 1.9, 1.0, "Linear C\n(no bias)", LINEAR)
    box(ax, 17.9, 7.7, 2.0, 1.4, "+ D·u\n(skip)\n= y", OTHER)

    arrow(ax, (1.8, 6.35), (2.75, 7.9))
    arrow(ax, (1.8, 5.65), (2.75, 4.1))
    arrow(ax, (4.65, 8.2), (5.25, 8.2))
    arrow(ax, (4.65, 3.8), (5.25, 3.8))
    arrow(ax, (7.35, 8.2), (8.3, 6.55))
    arrow(ax, (7.35, 3.8), (8.3, 5.45))
    arrow(ax, (10.15, 6.0), (10.85, 6.0))
    arrow(ax, (13.55, 6.0), (14.75, 6.0))
    arrow(ax, (15.4, 6.45), (15.4, 8.2))
    arrow(ax, (16.35, 8.7), (16.9, 8.0))

    ax.text(12.2, 3.2,
            "the sequential loop (parallel=False) computes the identical\n"
            "recursion x_k = a_k·x_{k-1} + b_k one step at a time --\n"
            "example.py checks the two paths agree to ~1e-7",
            fontsize=8.5, ha="center", color="#475569", style="italic")

    ax.set_title("LrcSSM: whole sequence solved in one call", fontsize=11)
```

## How the two solve paths compare

The scan trades sequential *depth* for more total work (it's not
asymptotically cheaper in FLOPs, just far more parallelizable), which is
what lets it run fast on a GPU/TPU even though it does more arithmetic than
the loop. The plot below is a purely structural illustration of that
depth trade-off (not a wall-clock benchmark -- run
{doc}`../benchmarks` yourself for actual timings on your hardware):

```{eval-rst}
.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    T = np.arange(1, 2049)
    sequential_depth = T
    parallel_depth = np.ceil(np.log2(T))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(T, sequential_depth, label="sequential loop:  O(T)")
    ax.plot(T, parallel_depth, label="parallel scan:  O(log T)")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("sequential steps required")
    ax.set_yscale("log")
    ax.set_title("Sequential depth: loop vs. diagonal parallel scan")
    ax.legend()
```

## Try it

```bash
python models/lrcssm/example.py --device auto     # ETTh1 forecasting
```

or open [`models/lrcssm/example.ipynb`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/lrcssm/example.ipynb),
which also times the sequential vs. parallel paths. Full runnable code:
[`models/lrcssm/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/lrcssm/model.py) ·
[`models/lrcssm/README.md`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/lrcssm/README.md).

## References

```{eval-rst}
.. bibliography::
   :filter: docname in docnames
```
