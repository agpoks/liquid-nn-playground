"""Liquid-LSTM -- an LSTM with an input-gated (liquid) continuous-time cell state.

References: Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural
Computation, 1997 (the LSTM gating this builds on); Hasani, Lechner, Amini,
Rus, Grosu, "Liquid Time-constant Networks", AAAI 2021 (the continuous-time,
input-gated leak this adds). See papers/README.md.

This is NOT a reimplementation of a single paper -- unlike every other model
in models/, there is no "Liquid-LSTM" paper. It's a hybrid built for this
repo to answer the question "what would combining LSTM's gating with LTC's
continuous time constant look like?", clearly labeled as such.

Standard LSTM computes four gates and a discrete cell-state update:

    f_t = sigmoid(W_f [x_t, h_prev] + b_f)   # forget gate
    i_t = sigmoid(W_i [x_t, h_prev] + b_i)   # input gate
    g_t = tanh(W_g [x_t, h_prev] + b_g)      # candidate
    o_t = sigmoid(W_o [x_t, h_prev] + b_o)   # output gate
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * tanh(c_t)

Liquid-LSTM keeps all four gates (so it still has LSTM's separate write/erase/
read control, unlike LTC's single synapse f), but replaces the discrete cell
update with LTC's continuous-time ODE, using the forget gate to build an
input-dependent effective time constant exactly the way LTC's f_i does:

    dc/dt = -c / tau_eff  +  i_t * g_t,     1/tau_eff = 1/tau + (1 - f_t)

f_t near 1 ("remember") pushes (1 - f_t) near 0, so tau_eff stays close to
the free-running tau -- slow decay, long memory, same as standard LSTM's
forget gate near 1. f_t near 0 ("forget") pushes tau_eff down towards
1/(1/tau + 1) -- fast decay, same qualitative behavior as standard LSTM's
forget gate near 0, but now realized as a genuinely continuous-time leak
(with its own ode_unfolds sub-steps and dt-awareness) instead of one
discrete multiply. Integrated with the same fused semi-implicit Euler
scheme as models/ltc and models/ctrnn.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LiquidLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, ode_unfolds: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        in_dim = input_size + hidden_size

        self.forget_gate = nn.Linear(in_dim, hidden_size)
        self.input_gate = nn.Linear(in_dim, hidden_size)
        self.candidate = nn.Linear(in_dim, hidden_size)
        self.output_gate = nn.Linear(in_dim, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, dt: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([x_t, h_prev], dim=-1)
        f = torch.sigmoid(self.forget_gate(z))
        i = torch.sigmoid(self.input_gate(z))
        g = torch.tanh(self.candidate(z))
        o = torch.sigmoid(self.output_gate(z))

        tau = nn.functional.softplus(self.tau) + 1e-3
        sub_dt = dt / self.ode_unfolds
        drive = i * g  # frozen for the sub-steps below, like LTC freezes f*A per input step

        c = c_prev
        for _ in range(self.ode_unfolds):
            numerator = c + sub_dt * drive
            denominator = 1.0 + sub_dt * (1.0 / tau + (1.0 - f))
            c = numerator / denominator

        h = o * torch.tanh(c)
        return h, c


class LiquidLSTMModel(nn.Module):
    """Wraps LiquidLSTMCell into a full sequence model: (B, T, input_size) -> (B, out_size)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, ode_unfolds: int = 6):
        super().__init__()
        self.cell = LiquidLSTMCell(input_size, hidden_size, ode_unfolds=ode_unfolds)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
        return self.readout(h)
