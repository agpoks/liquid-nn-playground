"""Continuous-time RNN (CT-RNN) -- an ODE-defined RNN with a *fixed* time constant.

Reference: Funahashi & Nakamura, "Approximation of dynamical systems by
continuous time recurrent neural networks", Neural Networks, 1993.
See papers/README.md.

Each hidden unit obeys:

    dh/dt = (-h + tanh(W_x x + W_h h + b)) / tau

with tau a learnable per-unit time constant that is fixed across inputs --
it never sees x or h at inference time, unlike LTC's f_i. This is the direct
ancestor models/ltc generalizes: LTC's
`dx/dt = -x/tau + f(x, I) * (A - x)` is exactly this equation's family with
the leak additionally gated by an input-dependent synapse f_i, so the
*effective* time constant becomes 1 / (1/tau_i + f_i(x, I)) instead of the
constant tau used here. CT-RNN is what you get by freezing that gate at a
constant -- there is no "liquid" property (no dependence of the dynamics'
speed on the current input) in this module at all.

Integrated here with the exact same fused semi-implicit (backward) Euler
scheme models/ltc/model.py uses (same `ode_unfolds`, same unconditional
stability), so the only code difference from `LTCCell.forward` is what the
frozen per-substep target is a function of.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CTRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, ode_unfolds: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        tau = nn.functional.softplus(self.tau) + 1e-3
        sub_dt = dt / self.ode_unfolds

        h = h_prev
        for _ in range(self.ode_unfolds):
            target = torch.tanh(self.w_in(x_t) + self.w_rec(h))  # frozen for this sub-step, like LTC freezes f
            h = (h + (sub_dt / tau) * target) / (1.0 + sub_dt / tau)
        return h


class CTRNNModel(nn.Module):
    """Wraps CTRNNCell into a full sequence model: (B, T, input_size) -> (B, out_size)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, ode_unfolds: int = 6):
        super().__init__()
        self.cell = CTRNNCell(input_size, hidden_size, ode_unfolds=ode_unfolds)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.readout(h)
