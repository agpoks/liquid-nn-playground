"""Generic train/eval loop shared by every (model, dataset) combo in benchmarks/."""

from __future__ import annotations

import time

import torch

from liquid_playground.benchmark.metrics import compute_metric, loss_fn_for
from liquid_playground.benchmark.registry import DATASET_REGISTRY, MODEL_REGISTRY
from liquid_playground.utils.seed import set_seed


def run_benchmark(
    model_name: str,
    dataset_name: str,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    model_kwargs: dict | None = None,
    dataset_kwargs: dict | None = None,
) -> dict:
    set_seed(seed)
    model_kwargs = model_kwargs or {}
    dataset_kwargs = dataset_kwargs or {}

    loader, task_type, fixed_output_size = DATASET_REGISTRY[dataset_name]
    train_x, train_y, test_x, test_y = loader(**dataset_kwargs)
    train_x, test_x = train_x.to(device), test_x.to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)

    output_size = fixed_output_size or (
        int(train_y.max().item()) + 1 if task_type == "multiclass" else train_y.shape[-1]
    )

    factory = MODEL_REGISTRY[model_name]
    model = factory(input_size=train_x.shape[-1], output_size=output_size, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn_for(task_type)

    n_train = train_x.shape[0]
    t0 = time.perf_counter()
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            out = model(train_x[idx])
            loss = loss_fn(out, train_y[idx])
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        test_metric = compute_metric(task_type, model(test_x), test_y)

    return {
        "model": model_name,
        "dataset": dataset_name,
        "task_type": task_type,
        "params": n_params,
        "epochs": epochs,
        "train_time_s": round(train_time, 2),
        "metric_name": "accuracy" if task_type != "regression" else "mse",
        "metric": round(test_metric, 4),
        "device": str(device),
    }
