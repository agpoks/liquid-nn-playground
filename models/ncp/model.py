"""Neural Circuit Policies (NCP): sparse, structured wiring over liquid neurons.

Reference: Lechner, Hasani, Amini, Henzinger, Rus, Grosu,
"Neural circuit policies enabling auditable autonomy", Nature Machine
Intelligence 2020. See papers/README.md.

Instead of a fully-connected recurrent layer, NCP wires liquid neurons
(here: LTC neurons, as in the original paper) into four feed-forward-ish
layers loosely modeled on the C. elegans nervous system:

    sensory (=raw input) -> inter -> command (recurrent) -> motor (=output)

Every projection is *sparse* (each source neuron fans out to only a handful
of targets, chosen randomly at construction time and fixed thereafter as a
binary mask). This cuts parameter count drastically vs. a dense recurrent
layer of the same neuron count and, per the paper, makes the resulting policy
easier to interpret (you can trace which sensory neurons influence which
motor neuron).

This is a compact educational reimplementation of the wiring + LTC dynamics;
for the full, highly configurable version (AutoNCP, custom wiring diagrams,
TF/PyTorch parity) see https://github.com/mlech26l/ncps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NCPWiring:
    """Builds fixed sparse binary masks for the sensory->inter->command->motor graph."""

    def __init__(
        self,
        input_size: int,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_fanout: int = 4,
        inter_fanout: int = 4,
        command_recurrent_fanout: int = 4,
        motor_fanin: int = 6,
        seed: int = 0,
    ):
        self.input_size = input_size
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.total_units = inter_neurons + command_neurons + motor_neurons

        g = torch.Generator().manual_seed(seed)

        # global index ranges within the hidden-state vector
        self._inter = slice(0, inter_neurons)
        self._command = slice(inter_neurons, inter_neurons + command_neurons)
        self._motor = slice(inter_neurons + command_neurons, self.total_units)

        self.input_mask = self._sparse_mask((self.total_units, input_size), self._inter, sensory_fanout, g)
        rec = torch.zeros(self.total_units, self.total_units)
        rec += self._sparse_block((inter_neurons, command_neurons), inter_fanout, g, self._inter, self._command)
        rec += self._sparse_block(
            (command_neurons, command_neurons), command_recurrent_fanout, g, self._command, self._command
        )
        rec += self._sparse_block_fanin((command_neurons, motor_neurons), motor_fanin, g, self._command, self._motor)
        self.recurrent_mask = rec.clamp(max=1.0)

    @staticmethod
    def _sparse_mask(shape, target_rows: slice, fanout: int, g: torch.Generator) -> torch.Tensor:
        mask = torch.zeros(shape)
        n_targets = target_rows.stop - target_rows.start
        n_sources = shape[1]
        for src in range(n_sources):
            chosen = torch.randperm(n_targets, generator=g)[: min(fanout, n_targets)]
            mask[target_rows.start + chosen, src] = 1.0
        return mask

    def _sparse_block(self, shape, fanout, g, src_slice, tgt_slice):
        """shape = (n_src, n_tgt); each source fans out to `fanout` random targets."""
        n_src, n_tgt = shape
        block = torch.zeros(self.total_units, self.total_units)
        for src in range(n_src):
            chosen = torch.randperm(n_tgt, generator=g)[: min(fanout, n_tgt)]
            block[tgt_slice.start + chosen, src_slice.start + src] = 1.0
        return block

    def _sparse_block_fanin(self, shape, fanin, g, src_slice, tgt_slice):
        """shape = (n_src, n_tgt); each *target* draws from `fanin` random sources."""
        n_src, n_tgt = shape
        block = torch.zeros(self.total_units, self.total_units)
        for tgt in range(n_tgt):
            chosen = torch.randperm(n_src, generator=g)[: min(fanin, n_src)]
            block[tgt_slice.start + tgt, src_slice.start + chosen] = 1.0
        return block


class NCPCell(nn.Module):
    """LTC dynamics restricted to the sparse NCPWiring connectivity."""

    def __init__(self, wiring: NCPWiring, ode_unfolds: int = 6):
        super().__init__()
        self.wiring = wiring
        self.ode_unfolds = ode_unfolds
        units = wiring.total_units

        self.w_rec = nn.Parameter(torch.randn(units, units) * 0.3)
        self.w_in = nn.Parameter(torch.randn(units, wiring.input_size) * 0.3)
        self.bias = nn.Parameter(torch.zeros(units))
        self.tau = nn.Parameter(torch.ones(units))
        self.A = nn.Parameter(torch.zeros(units))

        self.register_buffer("rec_mask", wiring.recurrent_mask)
        self.register_buffer("in_mask", wiring.input_mask)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        tau = nn.functional.softplus(self.tau) + 1e-3
        sub_dt = dt / self.ode_unfolds
        w_rec = self.w_rec * self.rec_mask
        w_in = self.w_in * self.in_mask

        h = h_prev
        for _ in range(self.ode_unfolds):
            pre = h @ w_rec.T + x_t @ w_in.T + self.bias
            f = torch.sigmoid(pre)
            numerator = h + sub_dt * f * self.A
            denominator = 1.0 + sub_dt * (1.0 / tau + f)
            h = numerator / denominator
        return h


class NCPModel(nn.Module):
    """(B, T, input_size) -> (B, output_size), output read from the motor neurons."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        inter_neurons: int = 16,
        command_neurons: int = 12,
        motor_neurons: int = 8,
        seed: int = 0,
    ):
        super().__init__()
        self.wiring = NCPWiring(input_size, inter_neurons, command_neurons, motor_neurons, seed=seed)
        self.cell = NCPCell(self.wiring)
        self.readout = nn.Linear(motor_neurons, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.wiring.total_units, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        motor = h[:, self.wiring._motor]
        return self.readout(motor)
