"""Single-seed metrics for 2D runs."""

from __future__ import annotations

import torch
from torch import Tensor

from .energy import EnergyClassifier
from .models import FlowNet


@torch.no_grad()
def compute_mmd(x: Tensor, y: Tensor, sigma: float = 1.0, max_points: int = 1000) -> float:
    count = min(len(x), len(y), max_points)
    x, y = x[:count].float(), y[:count].float()

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        distances = torch.cdist(a, b).square()
        return torch.exp(-distances / (2 * sigma**2))

    return float((kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()).cpu())


@torch.no_grad()
def classifier_metrics(classifier: EnergyClassifier, generated: Tensor, device: torch.device) -> dict[str, float]:
    classifier.to(device).eval()
    retain_probability = torch.sigmoid(classifier(generated.to(device)))
    retention_rate = float((retain_probability >= 0.5).float().mean().cpu())
    forget_rate = 1.0 - retention_rate
    leakage = float((1.0 - retain_probability).mean().cpu())
    return {
        "retention_rate": retention_rate,
        "forget_rate": forget_rate,
        "leakage": leakage,
    }


def compute_path_length(trajectory: Tensor) -> float:
    increments = trajectory[1:] - trajectory[:-1]
    return float(torch.linalg.vector_norm(increments, dim=-1).mean(dim=1).sum())


@torch.no_grad()
def compute_vector_field_norm(
    model: FlowNet,
    *,
    device: torch.device,
    resolution: int = 50,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
    time: float = 0.5,
) -> float:
    x, y = torch.meshgrid(
        torch.linspace(*xlim, resolution),
        torch.linspace(*ylim, resolution),
        indexing="xy",
    )
    grid = torch.stack((x, y), dim=-1).reshape(-1, 2).to(device)
    times = torch.full((len(grid), 1), time, device=device)
    model.to(device).eval()
    velocity = model(grid, times)
    return float(velocity.square().sum(dim=1).mean().cpu())


def evaluate_run(
    classifier: EnergyClassifier,
    unlearning_model: FlowNet,
    trajectory: Tensor,
    retained_data: Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    generated = trajectory[-1]
    metrics = classifier_metrics(classifier, generated, device)
    metrics.update(
        {
            "mmd_to_retain": compute_mmd(generated, retained_data),
            "path_length": compute_path_length(trajectory),
            "vector_field_norm_t0.5": compute_vector_field_norm(unlearning_model, device=device),
        }
    )
    return metrics

