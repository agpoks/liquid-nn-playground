"""CfC in JAX/Flax -- same architecture as model.py, different framework.

Included to make it easy to compare a liquid model's PyTorch vs. JAX
performance/ergonomics without re-deriving the math: this is line-for-line
the same closed-form gated update as CfCCell/CfCModel in model.py (see that
file's docstring, and papers/README.md, for the math). Only CfC gets a JAX
port for now -- see models/cfc/README.md for the reasoning.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class CfCCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, h_prev: jnp.ndarray, dt: float = 1.0) -> jnp.ndarray:
        z = nn.tanh(nn.Dense(self.hidden_size, name="backbone")(jnp.concatenate([x_t, h_prev], axis=-1)))
        f = nn.Dense(self.hidden_size, name="f_head")(z)
        g = nn.tanh(nn.Dense(self.hidden_size, name="g_head")(z))
        h_proj = nn.tanh(nn.Dense(self.hidden_size, name="h_head")(z))
        time_scale = self.param("time_scale", nn.initializers.ones, (self.hidden_size,))

        gate = nn.sigmoid(-f * (jnp.abs(time_scale) + 1e-3) * dt)
        return gate * h_proj + (1 - gate) * g


class CfCModel(nn.Module):
    """(B, T, input_size) -> (B, output_size). Same interface shape as the PyTorch CfCModel."""

    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        cell = CfCCell(self.hidden_size)
        h = jnp.zeros((batch, self.hidden_size), dtype=x.dtype)
        for t in range(seq_len):
            h = cell(x[:, t, :], h)
        return nn.Dense(self.output_size, name="readout")(h)
