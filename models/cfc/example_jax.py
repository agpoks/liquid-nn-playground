"""JAX/Flax port of example.py -- same CfC architecture, same dataset, same
task, so you can compare the PyTorch and JAX versions directly.

    python models/cfc/example_jax.py --device auto --epochs 30

Requires the 'jax' extra: pip install -e ".[jax]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from liquid_playground.data import load_person_activity  # noqa: E402
from model_jax import CfCModel  # noqa: E402


def resolve_jax_device(name: str) -> jax.Device:
    """JAX's own analogue of liquid_playground.device.resolve_device (that one
    is torch-specific). 'auto' picks JAX's default backend (GPU/TPU if JAX was
    built with one and one is visible, else CPU); otherwise ask for that
    platform explicitly."""
    if name == "auto":
        return jax.devices()[0]
    try:
        return jax.devices(name)[0]
    except RuntimeError as e:
        raise RuntimeError(
            f"--device {name} requested but no JAX '{name}' backend/device is available. "
            "Use --device cpu or --device auto."
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu", "tpu"],
        help="Compute device. 'auto' (default) uses JAX's default backend, else CPU.",
    )
    args = parser.parse_args()

    device = resolve_jax_device(args.device)
    print(f"Using device: {device}")

    with jax.default_device(device):
        train_x_t, train_y_t, test_x_t, test_y_t = load_person_activity()
        train_x = jnp.asarray(train_x_t.numpy())
        train_y = jnp.asarray(train_y_t.numpy())
        test_x = jnp.asarray(test_x_t.numpy())
        test_y = jnp.asarray(test_y_t.numpy())
        n_classes = int(train_y.max()) + 1

        model = CfCModel(hidden_size=args.hidden_size, output_size=n_classes)
        key = jax.random.PRNGKey(args.seed)
        params = model.init(key, train_x[:1])["params"]

        opt = optax.adam(args.lr)
        opt_state = opt.init(params)

        def loss_fn(params, x, y):
            logits = model.apply({"params": params}, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        @jax.jit
        def train_step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        @jax.jit
        def eval_acc(params, x, y):
            logits = model.apply({"params": params}, x)
            return (jnp.argmax(logits, axis=-1) == y).mean()

        for epoch in range(1, args.epochs + 1):
            params, opt_state, loss = train_step(params, opt_state, train_x, train_y)

            if epoch % 5 == 0 or epoch == args.epochs:
                test_acc = float(eval_acc(params, test_x, test_y))
                print(f"epoch {epoch:3d} | train_loss {float(loss):.4f} | test_acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
