"""Vanilla (Elman) RNN -- the simplest possible discrete-time baseline.

Reference: Elman, "Finding Structure in Time", Cognitive Science, 1990.
See papers/README.md.

    h_t = tanh(W_x x_t + W_h h_{t-1} + b)

There is no notion of elapsed time here at all: h_t depends only on the
*order* the inputs arrive in, not on how much wall-clock time separates them
-- no per-unit time constant, no gate, no ODE, no `dt`. This is the baseline
every other model in this repo implicitly improves on: see
models/ctrnn (adds a fixed time constant), models/node (adds a fully general
learned ODE), and models/ltc (adds an *input-dependent* time constant, the
point where "liquid" starts) for what each additional piece of machinery
buys over this two-line update.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        del dt  # accepted for interface parity with the other cells; unused -- no time-awareness at all
        return torch.tanh(self.w_in(x_t) + self.w_rec(h_prev))


class RNNModel(nn.Module):
    """Wraps RNNCell into a full sequence model: (B, T, input_size) -> (B, out_size)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.readout(h)
