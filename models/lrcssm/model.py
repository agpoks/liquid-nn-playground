"""LrcSSM: Liquid-Resistance Liquid-Capacitance State-Space Model.

Reference: Farsang, Hasani, Rus, Grosu, "Parallelization of Non-linear
State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks
for Efficient Sequence Modeling", NeurIPS 2025. arXiv:2505.21717.
See papers/README.md.

LrcSSM is a *non-linear* recurrent model (like LTC: state-dependent,
input-dependent dynamics) that is nonetheless trainable as fast as a linear
SSM. The trick: by construction the Jacobian of the state update w.r.t. the
previous state is forced to be *diagonal*, so the whole sequence can be solved
with a parallel (associative) scan in O(log T) sequential depth instead of a
plain O(T) sequential loop -- while retaining a formal gradient-stability
guarantee that purely linear input-dependent SSMs (e.g. Liquid-S4, Mamba)
don't provide.

The discrete-time update this module implements (paper Eq. 3-6, RC-circuit
form): each state unit behaves like an RC circuit whose resistance/capacitance
are themselves gated by the input, giving a diagonal, input-dependent decay
`a_k = sigmoid(-g(u_k))` and drive `b_k = g(u_k) * c(u_k)`:

    x_k = a_k * x_{k-1} + b_k

Because `a_k` only multiplies the *same* state index (never mixes indices),
the recurrence is an elementwise linear recursion in `k` and can be solved
with `torch.logcumsumexp`-style parallel scan. This module ships both a
sequential loop (default, easiest to read) and a parallel-scan path
(`parallel=True`) that produces identical results.

This is a compact educational reimplementation of the core diagonal-Jacobian
idea, not the paper's optimized CUDA parallel-scan kernel -- for that, see
https://github.com/MoniFarsang/LrcSSM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _parallel_diagonal_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve x_k = a_k * x_{k-1} + b_k (x_0 = b_0) for all k in parallel.

    a, b: (B, T, D). Returns x: (B, T, D).

    Hillis-Steele associative scan over the composition operator
    (A1, B1) . (A2, B2) = (A2*A1, A2*B1 + B2), which represents chaining two
    affine maps. After an inclusive scan, position k holds (A_k, B_k) with
    x_k = A_k * x_{-1} + B_k = B_k (since x_{-1} = 0) -- log2(T) sequential
    doubling steps, each pure multiply/add, so unlike a cumulative-product
    change of variables this never divides by a (possibly tiny) running
    product and stays numerically close to the sequential loop.
    """
    seq_len = a.shape[1]
    a, b = a.clone(), b.clone()
    offset = 1
    while offset < seq_len:
        a_prev = torch.cat([torch.ones_like(a[:, :offset]), a[:, :-offset]], dim=1)
        b_prev = torch.cat([torch.zeros_like(b[:, :offset]), b[:, :-offset]], dim=1)
        b = a * b_prev + b
        a = a * a_prev
        offset *= 2
    return b


class LrcSSMLayer(nn.Module):
    def __init__(self, input_size: int, state_size: int):
        super().__init__()
        self.state_size = state_size
        # "resistance" gate g(u) and "capacitance"/candidate c(u) -- both
        # input-dependent, giving the diagonal, liquid RC-circuit dynamics.
        self.g_head = nn.Linear(input_size, state_size)
        self.c_head = nn.Linear(input_size, state_size)
        self.C = nn.Linear(state_size, input_size, bias=False)
        self.D = nn.Parameter(torch.zeros(input_size))

    def forward(self, u: torch.Tensor, parallel: bool = True) -> torch.Tensor:
        """u: (B, T, input_size) -> y: (B, T, input_size)."""
        g = torch.sigmoid(self.g_head(u))  # (B, T, D) in (0, 1): "conductance"
        c = torch.tanh(self.c_head(u))  # (B, T, D): candidate level
        a = 1.0 - g  # diagonal decay: a_k in (0, 1) => unconditionally stable
        b = g * c

        if parallel:
            x = _parallel_diagonal_scan(a, b)
        else:
            batch, seq_len, _ = u.shape
            x_t = torch.zeros(batch, self.state_size, device=u.device, dtype=u.dtype)
            xs = []
            for t in range(seq_len):
                x_t = a[:, t, :] * x_t + b[:, t, :]
                xs.append(x_t)
            x = torch.stack(xs, dim=1)

        return self.C(x) + self.D * u


class LrcSSMModel(nn.Module):
    """(B, T, input_size) -> (B, output_size), pooled over time."""

    def __init__(self, input_size: int, state_size: int, output_size: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([LrcSSMLayer(input_size, state_size) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(input_size) for _ in range(n_layers)])
        self.readout = nn.Linear(input_size, output_size)

    def forward(self, u: torch.Tensor, parallel: bool = True) -> torch.Tensor:
        x = u
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x, parallel=parallel))
        pooled = x.mean(dim=1)
        return self.readout(pooled)
