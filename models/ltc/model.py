"""Liquid Time-Constant (LTC) network.

Reference: Hasani, Lechner, Amini, Rus, Grosu — "Liquid Time-constant Networks",
AAAI 2021. arXiv:2006.04439. See papers/README.md.

Each hidden unit obeys the ODE (paper Eq. 5-6):

    dx_i/dt = -x_i / tau_i + f_i(x, I; theta) * (A_i - x_i)

where f_i is a sigmoidal synapse driven by the recurrent state and the input,
tau_i is a learnable time constant, and A_i is a learnable reversal potential.
The "liquid" property is that the *effective* time constant,
1 / (1/tau_i + f_i(x, I)), varies with the input at every step instead of being
fixed -- the network speeds up or slows down its own dynamics based on what it
is looking at.

This module integrates that ODE with the fused semi-implicit (backward) Euler
scheme from the paper/official implementation: f is evaluated explicitly at the
current state, then the resulting *linear* ODE in x is solved implicitly, which
is unconditionally stable and lets us take a handful of sub-steps per input
without a general-purpose ODE solver. This is a compact educational
reimplementation; for the exact, highly-optimized reference version see
https://github.com/mlech26l/ncps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LTCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, ode_unfolds: int = 6):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds

        # synapse weights: hidden state + input -> pre-activation of f_i
        self.w_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.w_in = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # learnable per-neuron time constant (softplus'd to stay positive) and
        # reversal potential A
        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.A = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        tau = nn.functional.softplus(self.tau) + 1e-3
        sub_dt = dt / self.ode_unfolds

        h = h_prev
        for _ in range(self.ode_unfolds):
            pre = h @ self.w_rec.T + x_t @ self.w_in.T + self.bias
            f = torch.sigmoid(pre)  # synaptic activation, frozen for this sub-step
            numerator = h + sub_dt * f * self.A
            denominator = 1.0 + sub_dt * (1.0 / tau + f)
            h = numerator / denominator
        return h


class LTCModel(nn.Module):
    """Wraps LTCCell into a full sequence model: (B, T, input_size) -> (B, out_size)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, ode_unfolds: int = 6):
        super().__init__()
        self.cell = LTCCell(input_size, hidden_size, ode_unfolds=ode_unfolds)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.readout(h)
