"""Closed-form Continuous-time (CfC) network.

Reference: Hasani, Lechner, Amini, Ray, Chahine, Wang, Rus,
"Closed-form Continuous-time Neural Networks", Nature Machine Intelligence 2022.
arXiv:2106.13898. See papers/README.md.

CfC replaces LTC's numerically-integrated ODE with a *closed-form* approximation
of its solution (paper Eq. 8-10), so a full sequence rollout is a handful of
matrix ops instead of an unrolled ODE solver -- the paper reports >100x
training/inference speedups over ODE-based liquid networks with comparable
accuracy. The closed-form state update used here is:

    h(t) = sigmoid(-f(x) * t) * g(x) + (1 - sigmoid(-f(x) * t)) * h(x)

where f, g, h are small feed-forward "backbone" heads shared across time, and
t is the (learnable-scaled) elapsed time between steps. sigmoid(-f(x)*t) plays
the role of the input-dependent liquid time-constant gate: large f(x) collapses
towards the new candidate g(x) quickly, small f(x) preserves the old state h(x).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CfCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        in_dim = input_size + hidden_size

        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.Tanh())
        self.f_head = nn.Linear(hidden_size, hidden_size)  # gate / effective rate
        self.g_head = nn.Linear(hidden_size, hidden_size)  # candidate state
        self.h_head = nn.Linear(hidden_size, hidden_size)  # retained-state projection
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        z = self.backbone(torch.cat([x_t, h_prev], dim=-1))
        f = self.f_head(z)
        g = torch.tanh(self.g_head(z))
        h_proj = torch.tanh(self.h_head(z))

        gate = torch.sigmoid(-f * (self.time_scale.abs() + 1e-3) * dt)
        h_new = gate * h_proj + (1 - gate) * g
        return h_new


class CfCModel(nn.Module):
    """Wraps CfCCell into a full sequence model: (B, T, input_size) -> (B, out_size)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.readout(h)
