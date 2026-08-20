"""Liquid-S4: a linear liquid time-constant state-space model.

Reference: Hasani, Lechner, Wang, Chahine, Amini, Rus,
"Liquid Structural State-Space Models", ICLR 2023. arXiv:2209.12951.
See papers/README.md.

Liquid-S4 takes a structured state-space model (S4-style: a diagonal linear
recurrence x_k = A x_{k-1} + B u_k, y_k = C x_k) and makes the state
transition *input-dependent*, the way LTC/CfC make their time constants
input-dependent -- the paper does this with a low-rank correction of A driven
by the input.

This module is a compact, real-valued, sequential-scan reimplementation of
that idea (a diagonal HiPPO-inspired base transition `a`, gated per-step by a
learned function of the input) -- readable, but *not* the paper's FFT-parallel
convolutional training path or its complex-HiPPO initialization. For the exact,
state-of-the-art, LRA-benchmarked version see https://github.com/raminmh/liquid-s4.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LiquidS4Layer(nn.Module):
    """One diagonal SSM channel bank with an input-dependent ("liquid") decay gate."""

    def __init__(self, input_size: int, state_size: int):
        super().__init__()
        self.state_size = state_size

        # HiPPO-inspired init: negative real decay rates spread on a log scale,
        # so different state units naturally specialize to different timescales.
        log_decay = torch.linspace(math.log(0.5), math.log(0.001), state_size)
        self.log_decay = nn.Parameter(log_decay)  # a_base = -exp(log_decay), in (-0.5, -0.001)

        self.B = nn.Linear(input_size, state_size, bias=False)
        self.C = nn.Linear(state_size, input_size, bias=False)
        self.D = nn.Parameter(torch.zeros(input_size))

        # liquid gate: input-dependent multiplicative correction to the decay
        self.liquid_gate = nn.Linear(input_size, state_size)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, T, input_size) -> y: (B, T, input_size)."""
        batch, seq_len, _ = u.shape
        a_base = -torch.exp(self.log_decay)  # (state_size,), always < 0 => stable

        x = torch.zeros(batch, self.state_size, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(seq_len):
            u_t = u[:, t, :]
            liquid_mod = torch.sigmoid(self.liquid_gate(u_t))  # (B, state_size) in (0, 1)
            a_eff = torch.exp(a_base) * liquid_mod  # input-dependent effective decay
            x = a_eff * x + self.B(u_t)
            ys.append(self.C(x) + self.D * u_t)
        return torch.stack(ys, dim=1)


class LiquidS4Model(nn.Module):
    """(B, T, input_size) -> (B, output_size), pooled over time for classification/forecasting heads."""

    def __init__(self, input_size: int, state_size: int, output_size: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [LiquidS4Layer(input_size, state_size) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(input_size) for _ in range(n_layers)])
        self.readout = nn.Linear(input_size, output_size)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = u
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        pooled = x.mean(dim=1)
        return self.readout(pooled)
