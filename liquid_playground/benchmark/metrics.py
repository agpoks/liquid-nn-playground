from __future__ import annotations

import torch


def compute_metric(task_type: str, logits: torch.Tensor, y: torch.Tensor) -> float:
    if task_type == "binary":
        return ((logits.squeeze(-1) > 0).float() == y).float().mean().item()
    if task_type == "multiclass":
        return (logits.argmax(-1) == y).float().mean().item()
    if task_type == "regression":
        return torch.nn.functional.mse_loss(logits, y).item()
    raise ValueError(f"unknown task_type: {task_type}")


def loss_fn_for(task_type: str):
    if task_type == "binary":
        return lambda logits, y: torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), y)
    if task_type == "multiclass":
        return torch.nn.functional.cross_entropy
    if task_type == "regression":
        return torch.nn.functional.mse_loss
    raise ValueError(f"unknown task_type: {task_type}")
