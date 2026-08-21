"""Train CT-GRU on the UCI Ozone Level Detection dataset.

    python models/ctgru/example.py --device auto --epochs 30

Same task as models/ltc/example.py -- direct comparison. See model.py for
the multi-timescale memory-trace mechanism this completes in the RNN ->
CT-RNN -> CT-GRU -> Neural ODE/Liquid-LSTM -> LTC/CfC family.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from liquid_playground.data import load_ozone  # noqa: E402
from liquid_playground.device import add_device_arg, resolve_device  # noqa: E402
from liquid_playground.utils.seed import set_seed  # noqa: E402
from model import CTGRUModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-scales", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_x, train_y, test_x, test_y = load_ozone()
    train_x = train_x.unsqueeze(-1).to(device)  # (N, 72, 1)
    test_x = test_x.unsqueeze(-1).to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)

    model = CTGRUModel(
        input_size=1, hidden_size=args.hidden_size, output_size=1, num_scales=args.num_scales
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(train_x).squeeze(-1)
        loss = loss_fn(logits, train_y)
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_x).squeeze(-1)
                test_acc = ((test_logits > 0).float() == test_y).float().mean().item()
            print(f"epoch {epoch:3d} | train_loss {loss.item():.4f} | test_acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
