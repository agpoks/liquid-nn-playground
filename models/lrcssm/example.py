"""Train LrcSSM on ETTh1 long-horizon forecasting, comparing the sequential
loop and the parallel-scan solve (they should match to numerical precision).

    python models/lrcssm/example.py --device auto --epochs 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from liquid_playground.data import load_ett  # noqa: E402
from liquid_playground.device import add_device_arg, resolve_device  # noqa: E402
from liquid_playground.utils.seed import set_seed  # noqa: E402
from model import LrcSSMModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--state-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_x, train_y, test_x, test_y = load_ett(seq_len=args.seq_len, pred_len=args.pred_len)
    n_channels = train_x.shape[-1]
    train_x, test_x = train_x.to(device), test_x.to(device)
    train_y = train_y.reshape(train_y.shape[0], -1).to(device)
    test_y = test_y.reshape(test_y.shape[0], -1).to(device)

    model = LrcSSMModel(
        input_size=n_channels, state_size=args.state_size, output_size=args.pred_len * n_channels
    ).to(device)

    # sanity check: sequential and parallel-scan paths must agree
    with torch.no_grad():
        sample = train_x[:4]
        seq_out = model.layers[0](sample, parallel=False)
        par_out = model.layers[0](sample, parallel=True)
        max_diff = (seq_out - par_out).abs().max().item()
        print(f"sequential vs. parallel-scan max abs diff: {max_diff:.2e}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    batch_size = 64
    n_train = train_x.shape[0]
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(train_x[idx], parallel=True)
            loss = loss_fn(pred, train_y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        if epoch % 2 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                test_mse = loss_fn(model(test_x, parallel=True), test_y).item()
            print(f"epoch {epoch:3d} | train_mse {total_loss / n_train:.4f} | test_mse {test_mse:.4f}")

    print(f"total train time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
