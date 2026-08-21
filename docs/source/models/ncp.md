# NCP -- Neural Circuit Policies

NCP {cite}`lechner2020ncp` doesn't change the neuron dynamics at all -- it
reuses {doc}`ltc`'s ODE unchanged -- it changes the *wiring*. Instead of a
dense recurrent layer where every unit talks to every other unit, NCP
arranges liquid neurons into four sparse, structured layers loosely modeled
on the *C. elegans* nervous system:

$$
\text{sensory (input)} \;\longrightarrow\; \text{inter} \;\longrightarrow\; \text{command (recurrent)} \;\longrightarrow\; \text{motor (output)}
$$

## The equation

Per-neuron, the dynamics are the *identical* LTC ODE from {doc}`ltc`:

$$
\frac{dx_i}{dt} = -\left(\frac{1}{\tau_i} + f_i(x, I)\right)x_i + f_i(x, I)\,A_i
$$

The only change is that the weight matrices are masked to zero out every
connection that isn't allowed by the wiring diagram above. So NCP is best
understood as *LTC + a sparsity prior*, not a new dynamical system -- see
{doc}`../model_comparison`, where NCP and LTC share a row.

## How it's built: the wiring, not the neuron

`NCPWiring` in
[`models/ncp/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ncp/model.py)
builds two fixed binary masks at construction time (not learned): one for
sensory→inter connections (`input_mask`), one for the
inter→command→motor + command→command recurrent connections
(`recurrent_mask`). Each *source* neuron only fans out to a handful of
random targets (`sensory_fanout`, `inter_fanout`,
`command_recurrent_fanout` -- default 4), and motor neurons draw from a
random subset of command neurons (`motor_fanin`, default 6) rather than all
of them:

```python
self.input_mask = self._sparse_mask((self.total_units, input_size), self._inter, sensory_fanout, g)
rec  = self._sparse_block((inter_neurons, command_neurons), inter_fanout, g, self._inter, self._command)
rec += self._sparse_block((command_neurons, command_neurons), command_recurrent_fanout, g, self._command, self._command)
rec += self._sparse_block_fanin((command_neurons, motor_neurons), motor_fanin, g, self._command, self._motor)
```

`NCPCell.forward` is then line-for-line the same fused semi-implicit-Euler
LTC solve as {doc}`ltc`, just with the weight matrices multiplied by these
masks before use: `w_rec = self.w_rec * self.rec_mask`. `NCPModel` reads its
output from the **motor** neurons only (`h[:, self.wiring._motor]`), the
same way the biological wiring diagram routes only motor-neuron activity to
actuators.

Zoomed out, ignoring the per-neuron LTC math (see {doc}`ltc` for that), the
four-layer wiring itself looks like this:

```{eval-rst}
.. plot::

    from liquid_playground.utils.diagrams import new_ax, box, arrow, INPUT, STATE

    fig, ax = new_ax(figsize=(9.5, 4.4), xlim=(0, 15), ylim=(0, 7.5))

    box(ax, 1.2, 4.2, 1.8, 1.3, "sensory\n(input)", INPUT)
    box(ax, 4.4, 4.2, 1.8, 1.3, "inter\n16 units", STATE)
    box(ax, 7.9, 4.2, 2.0, 1.6, "command\n12 units", STATE)
    box(ax, 11.4, 4.2, 1.8, 1.3, "motor\n8 units", STATE)

    arrow(ax, (2.1, 4.2), (3.5, 4.2))
    ax.text(2.8, 4.75, "sparse\nfanout=4", fontsize=7.5, ha="center", color="#334155")
    arrow(ax, (5.3, 4.2), (6.9, 4.2))
    ax.text(6.1, 4.75, "sparse\nfanout=4", fontsize=7.5, ha="center", color="#334155")
    arrow(ax, (8.9, 4.2), (10.5, 4.2))
    ax.text(9.7, 4.75, "sparse\nfanin=6", fontsize=7.5, ha="center", color="#334155")
    arrow(ax, (7.4, 5.2), (8.4, 5.2), curve=-1.3, dashed=True)
    ax.text(7.9, 6.15, "sparse recurrent\nfanout=4", fontsize=7.5, ha="center", color="#334155")
    arrow(ax, (12.3, 4.2), (13.3, 4.2))
    ax.text(13.4, 4.2, "readout\n(Linear) → y", fontsize=8, va="center", color="#334155")

    ax.text(7.5, 1.6,
            "every unit above still runs the same LTC ODE from the LTC page --\n"
            "only the connectivity is masked to these sparse, one-way arrows",
            fontsize=8.5, ha="center", color="#475569", style="italic")

    ax.set_title("NCP wiring: sensory → inter → command → motor", fontsize=11)
```

## What the sparsity looks like

Each cell of the heatmap below is one *possible* recurrent connection
(target neuron on the vertical axis, source on the horizontal); black =
connected. Note the block structure: inter neurons (first block) only ever
appear as *sources* into command neurons, never as targets, and motor
neurons (last block) only ever appear as *targets* -- information flows
one-way through the layers even though the command block itself is
recurrent.

```{eval-rst}
.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    def sparse_block(n_src, n_tgt, fanout, mode="fanout"):
        block = np.zeros((n_tgt, n_src))
        if mode == "fanout":
            for src in range(n_src):
                chosen = rng.permutation(n_tgt)[: min(fanout, n_tgt)]
                block[chosen, src] = 1.0
        else:  # fanin
            for tgt in range(n_tgt):
                chosen = rng.permutation(n_src)[: min(fanout, n_src)]
                block[tgt, chosen] = 1.0
        return block

    inter, command, motor = 16, 12, 8
    total = inter + command + motor
    mask = np.zeros((total, total))
    mask[inter:inter + command, 0:inter] = sparse_block(inter, command, 4)
    mask[inter:inter + command, inter:inter + command] = sparse_block(command, command, 4)
    mask[inter + command:, inter:inter + command] = sparse_block(command, motor, 6, mode="fanin")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="Greys", aspect="auto")
    for boundary in [inter, inter + command]:
        ax.axhline(boundary - 0.5, color="tab:blue", linewidth=1)
        ax.axvline(boundary - 0.5, color="tab:blue", linewidth=1)
    ax.set_xlabel("source neuron  (inter | command | motor)")
    ax.set_ylabel("target neuron  (inter | command | motor)")
    ax.set_title("NCP sparse recurrent wiring mask")
```

## Try it

```bash
python models/ncp/example.py --device auto     # trains on UCI Room Occupancy
```

or open [`models/ncp/example.ipynb`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ncp/example.ipynb),
which also plots the actual wiring mask for a trained model. Full runnable
code: [`models/ncp/model.py`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ncp/model.py) ·
[`models/ncp/README.md`](https://github.com/agpoks/liquid-nn-playground/blob/main/models/ncp/README.md).

## References

```{eval-rst}
.. bibliography::
   :filter: docname in docnames
```
