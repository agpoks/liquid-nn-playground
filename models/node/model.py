"""Neural ODE -- a generic, unconstrained learned vector field for the hidden state.

Reference: Chen, Rubanova, Bettencourt, Duvenaud,
"Neural Ordinary Differential Equations", NeurIPS 2018. arXiv:1806.07366.
See papers/README.md.

    dh/dt = f_theta(h, x)

where f_theta is an arbitrary small MLP: no leak term, no fixed structure, no
built-in stability guarantee. models/ltc's
`dx/dt = -x/tau + f(x, I) * (A - x)` *is* a Neural ODE -- it is exactly this
equation with a specific, provably-bounded right-hand side substituted for
f_theta. This module is the unconstrained general case LTC specializes.

Because a fully generic f_theta has no linear structure to exploit for an
implicit solve the way LTC's does, this integrates with a hand-written
classic 4th-order Runge-Kutta (RK4) step instead of LTC's semi-implicit
Euler. Gradients here are obtained by autodiffing directly through the
unrolled RK4 steps rather than the paper's memory-efficient adjoint method --
fine at the sequence lengths used in this repo, but the adjoint trick (which
computes gradients by solving a second, backward-in-time ODE instead of
storing every solver step) is the paper's actual headline contribution and
is what makes Neural ODEs practical at much longer sequence lengths/depths;
see the paper Sec. 2-3 for the derivation. This reimplementation is a
compact educational one; for a production adjoint-method solver see
https://github.com/rtqichen/torchdiffeq (the paper authors' own package).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuralODECell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, ode_unfolds: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.f = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _dh(self, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.f(torch.cat([h, x_t], dim=-1))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        sub_dt = dt / self.ode_unfolds

        h = h_prev
        for _ in range(self.ode_unfolds):
            k1 = self._dh(h, x_t)
            k2 = self._dh(h + 0.5 * sub_dt * k1, x_t)
            k3 = self._dh(h + 0.5 * sub_dt * k2, x_t)
            k4 = self._dh(h + sub_dt * k3, x_t)
            h = h + (sub_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


class NeuralODEModel(nn.Module):
    """Wraps NeuralODECell into a full sequence model: (B, T, input_size) -> (B, out_size)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, ode_unfolds: int = 4):
        super().__init__()
        self.cell = NeuralODECell(input_size, hidden_size, ode_unfolds=ode_unfolds)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.readout(h)
