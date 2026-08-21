"""CT-GRU -- Continuous-Time Gated Recurrent Unit.

Reference: Mozer, Kazakov, Lindsey, "Discrete Event, Continuous Time RNNs",
2017. arXiv:1710.04110. See papers/README.md.

Completes the family the other baselines already sketch out: RNN (no time
axis) -> CT-RNN (one fixed time constant) -> CT-GRU (a *bank* of fixed
time constants, gated) -> Neural ODE / Liquid-LSTM -> LTC/CfC.

Where CT-RNN adds one learnable-but-fixed tau, CT-GRU instead keeps M
*pre-specified*, log-spaced time constants tau_tilde_1 < ... < tau_tilde_M
(hyperparameters, not learned) and, per unit, learns a soft distribution
over which of those M scales to read from and write to at every step. Each
unit's memory is really M parallel traces h_hat_i, one per scale, each
decaying at its own fixed rate -- "continuous time" here means an *exact*
exponential-decay closed form over the real elapsed time dt, not an
unrolled/approximate ODE solve like LTC/CT-RNN use.

Per step k, with elapsed time dt (paper's Delta t_k):

    ln(tau_R) = W_R [x_k, h_{k-1}] + b_R                    # (B, H) continuous retrieval-scale estimate
    r_i       = softmax_i( -(ln(tau_R) - ln(tau_tilde_i))^2 )   # (B, H, M) soft bin assignment

    ln(tau_S) = W_S [x_k, h_{k-1}] + b_S                    # (B, H) continuous storage-scale estimate
    s_i       = softmax_i( -(ln(tau_S) - ln(tau_tilde_i))^2 )   # (B, H, M)

    retrieved = sum_i r_i * h_hat_{k-1,i}                    # (B, H) -- read the traces back out
    q         = tanh(W_q [x_k, retrieved] + b_q)             # (B, H) -- GRU-style candidate value

    h_hat_{k,i} = [(1 - s_i) * h_hat_{k-1,i} + s_i * q] * exp(-dt / tau_tilde_i)   # (B, H, M)
    h_k         = sum_i h_hat_{k,i}                          # (B, H) -- readout

r_i/s_i are Gaussian-shaped softmax kernels centered on whichever fixed bin
tau_tilde_i is closest (in log-time) to the network's continuously-varying
estimate -- so a unit can smoothly slide which discrete scale it uses
without ever changing the M reference points themselves. Unlike LTC/CT-RNN,
there's no ode_unfolds loop: exp(-dt/tau_tilde_i) is the exact solution of
dh/dt = -h/tau_tilde_i, so decaying a trace by an elapsed time dt is one
closed-form multiply per scale, done in parallel across all M of them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CTGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_scales: int = 8,
        min_tau: float = 1.0,
        max_tau: float = 100.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_scales = num_scales

        log_tau_tilde = torch.linspace(math.log(min_tau), math.log(max_tau), num_scales)
        self.register_buffer("log_tau_tilde", log_tau_tilde)  # (M,), fixed -- not learned
        self.register_buffer("tau_tilde", torch.exp(log_tau_tilde))

        in_dim = input_size + hidden_size
        self.retrieval_gate = nn.Linear(in_dim, hidden_size)  # -> ln(tau_R), one estimate per unit
        self.storage_gate = nn.Linear(in_dim, hidden_size)  # -> ln(tau_S)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)  # q_k

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, hhat_prev: torch.Tensor, dt: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([x_t, h_prev], dim=-1)
        ln_tau_r = self.retrieval_gate(z).unsqueeze(-1)  # (B, H, 1)
        ln_tau_s = self.storage_gate(z).unsqueeze(-1)  # (B, H, 1)
        log_tau_tilde = self.log_tau_tilde.view(1, 1, -1)  # (1, 1, M)

        r = torch.softmax(-((ln_tau_r - log_tau_tilde) ** 2), dim=-1)  # (B, H, M)
        s = torch.softmax(-((ln_tau_s - log_tau_tilde) ** 2), dim=-1)  # (B, H, M)

        retrieved = (r * hhat_prev).sum(dim=-1)  # (B, H)
        q = torch.tanh(self.candidate(torch.cat([x_t, retrieved], dim=-1)))  # (B, H)

        decay = torch.exp(-dt / self.tau_tilde).view(1, 1, -1)  # (1, 1, M), exact closed-form decay
        hhat = ((1.0 - s) * hhat_prev + s * q.unsqueeze(-1)) * decay  # (B, H, M)
        h = hhat.sum(dim=-1)  # (B, H)
        return h, hhat


class CTGRUModel(nn.Module):
    """Wraps CTGRUCell into a full sequence model: (B, T, input_size) -> (B, out_size)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_scales: int = 8):
        super().__init__()
        self.cell = CTGRUCell(input_size, hidden_size, num_scales=num_scales)
        self.readout = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_scales = num_scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        hhat = torch.zeros(batch, self.hidden_size, self.num_scales, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h, hhat = self.cell(x[:, t, :], h, hhat)
        return self.readout(h)
