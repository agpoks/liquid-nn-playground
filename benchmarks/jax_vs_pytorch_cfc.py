"""Short JAX-vs-PyTorch benchmark: trains the *same* CfC architecture in both
frameworks (models/cfc/model.py vs models/cfc/model_jax.py -- line-for-line
the same closed-form update, see model.py's docstring) on the identical UCI
Person Activity task, and reports wall-clock time per training step and
final test accuracy side by side.

    python benchmarks/jax_vs_pytorch_cfc.py --device auto --epochs 15

Requires the 'jax' extra: pip install -e ".[jax]"

This is not a rigorous perf study (no warmup-excluded steady-state timing,
no multiple seeds) -- it's a quick, honest side-by-side of "the same model,
two frameworks, this machine" to sanity-check the JAX port's speed against
the PyTorch original.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "cfc"))

from liquid_playground.data import load_person_activity  # noqa: E402
from liquid_playground.device import resolve_device  # noqa: E402
from liquid_playground.utils.seed import set_seed  # noqa: E402
from model import CfCModel as CfCModelTorch  # noqa: E402


def run_pytorch(hidden_size: int, epochs: int, lr: float, seed: int, device_name: str) -> dict:
    set_seed(seed)
    device = resolve_device(device_name)

    train_x, train_y, test_x, test_y = load_person_activity()
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    n_classes = int(train_y.max().item()) + 1

    model = CfCModelTorch(input_size=train_x.shape[-1], hidden_size=hidden_size, output_size=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    step_times = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    model.eval()
    with torch.no_grad():
        test_acc = (model(test_x).argmax(-1) == test_y).float().mean().item()

    return {
        "framework": "pytorch",
        "device": str(device),
        "params": sum(p.numel() for p in model.parameters()),
        "avg_step_s": sum(step_times) / len(step_times),
        "first_step_s": step_times[0],
        "test_acc": test_acc,
    }


def run_jax(hidden_size: int, epochs: int, lr: float, seed: int, device_name: str) -> dict:
    import jax
    import jax.numpy as jnp
    import optax

    from example_jax import resolve_jax_device  # noqa: E402
    from model_jax import CfCModel as CfCModelJax  # noqa: E402

    jax_device_name = "auto" if device_name in ("auto", "cuda", "mps") else device_name
    device = resolve_jax_device(jax_device_name)

    with jax.default_device(device):
        train_x_t, train_y_t, test_x_t, test_y_t = load_person_activity()
        train_x = jnp.asarray(train_x_t.numpy())
        train_y = jnp.asarray(train_y_t.numpy())
        test_x = jnp.asarray(test_x_t.numpy())
        test_y = jnp.asarray(test_y_t.numpy())
        n_classes = int(train_y.max()) + 1

        model = CfCModelJax(hidden_size=hidden_size, output_size=n_classes)
        key = jax.random.PRNGKey(seed)
        params = model.init(key, train_x[:1])["params"]
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

        opt = optax.adam(lr)
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

        step_times = []
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            params, opt_state, loss = train_step(params, opt_state, train_x, train_y)
            loss.block_until_ready()  # JAX dispatch is async -- block for an honest per-step time
            step_times.append(time.perf_counter() - t0)

        test_acc = float(eval_acc(params, test_x, test_y))

    return {
        "framework": "jax",
        "device": str(device),
        "params": n_params,
        "avg_step_s": sum(step_times) / len(step_times),
        "first_step_s": step_times[0],
        "test_acc": test_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "gpu", "tpu"],
        help="Resolved per-framework: PyTorch sees cuda/mps/cpu, JAX sees gpu/tpu/cpu; "
        "'auto' picks each framework's own best available device.",
    )
    args = parser.parse_args()

    print(f"Person Activity, hidden_size={args.hidden_size}, epochs={args.epochs}, lr={args.lr}\n")

    torch_result = run_pytorch(args.hidden_size, args.epochs, args.lr, args.seed, args.device)
    print(f"[pytorch] {torch_result}")

    try:
        jax_result = run_jax(args.hidden_size, args.epochs, args.lr, args.seed, args.device)
        print(f"[jax]     {jax_result}")
    except ImportError:
        print("[jax]     skipped -- install the 'jax' extra: pip install -e \".[jax]\"")
        return

    print()
    header = f"{'framework':>10} | {'device':>8} | {'params':>8} | {'avg_step_s':>11} | {'first_step_s':>13} | {'test_acc':>8}"
    print(header)
    for r in (torch_result, jax_result):
        print(
            f"{r['framework']:>10} | {r['device']:>8} | {r['params']:>8} | "
            f"{r['avg_step_s']:>11.4f} | {r['first_step_s']:>13.4f} | {r['test_acc']:>8.3f}"
        )
    print(
        "\nfirst_step_s includes JIT/graph-compile time (JAX traces+compiles train_step "
        "on its first call); avg_step_s is dominated by that same one-time cost if --epochs is small."
    )


if __name__ == "__main__":
    main()
