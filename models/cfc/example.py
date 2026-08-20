"""Train a CfC network on the UCI Localization Data for Person Activity dataset.

    python models/cfc/example.py --device auto --epochs 30

This is one of the exact datasets used to benchmark CfC / NCP in the original
papers (see papers/README.md), so results here are directly comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from liquid_playground.data import load_person_activity  # noqa: E402
from liquid_playground.device import add_device_arg, resolve_device  # noqa: E402
from liquid_playground.utils.seed import set_seed  # noqa: E402
from model import CfCModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_x, train_y, test_x, test_y = load_person_activity()
    train_x, test_x = train_x.to(device), test_x.to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    n_classes = int(train_y.max().item()) + 1

    model = CfCModel(input_size=train_x.shape[-1], hidden_size=args.hidden_size, output_size=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                test_acc = (model(test_x).argmax(-1) == test_y).float().mean().item()
            print(f"epoch {epoch:3d} | train_loss {loss.item():.4f} | test_acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
