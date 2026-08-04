"""Flow matching optimization and deterministic trajectory sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .energy import EnergyClassifier, retention_weights
from .models import FlowNet


@dataclass(frozen=True)
class TrainingStats:
    final_loss: float
    optimizer_updates: int


def _loader(data: Tensor, batch_size: int, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(data.detach().cpu()),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def train_base_flow(
    model: FlowNet,
    target_data: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    progress: bool = True,
) -> TrainingStats:
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = _loader(target_data, batch_size, seed)
    updates, final_loss = 0, float("nan")
    iterator = tqdm(range(epochs), desc="Training Flow: Gaussian → R0", disable=not progress)
    for _ in iterator:
        for (target,) in loader:
            target = target.to(device)
            source = torch.randn_like(target)
            time = torch.rand(len(target), 1, device=device)
            interpolated = (1 - time) * source + time * target
            velocity = target - source
            loss = nn.functional.mse_loss(model(interpolated, time), velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            final_loss = float(loss.detach().cpu())
    model.eval()
    return TrainingStats(final_loss, updates)


def train_unlearning_flow(
    model: FlowNet,
    base_samples: Tensor,
    classifier: EnergyClassifier,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    suppression_lambda: float,
    device: torch.device,
    seed: int,
    progress: bool = True,
) -> TrainingStats:
    model.to(device).train()
    classifier.to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = _loader(base_samples, batch_size, seed)
    base_cpu = base_samples.detach().cpu()
    updates, final_loss = 0, float("nan")
    iterator = tqdm(range(epochs), desc="Training Flow: R0 → R̃", disable=not progress)
    for _ in iterator:
        for (source,) in loader:
            source = source.to(device)
            indices = torch.randint(0, len(base_cpu), (len(source),))
            target = base_cpu[indices].to(device)
            time = torch.rand(len(source), 1, device=device)
            interpolated = (1 - time) * source + time * target
            velocity = target - source
            with torch.no_grad():
                weights = retention_weights(classifier, target, suppression_lambda).reshape(-1, 1)
            prediction = model(interpolated, time)
            denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
            loss = (weights * (prediction - velocity).square()).sum() / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Unlearning loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            final_loss = float(loss.detach().cpu())
    model.eval()
    return TrainingStats(final_loss, updates)


@torch.no_grad()
def sample_base_flow(
    model: FlowNet,
    *,
    n_samples: int,
    integration_steps: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    model.to(device).eval()
    points = torch.randn(n_samples, model.dim, device=device)
    return _integrate(model, points, integration_steps, device)


@torch.no_grad()
def sample_from_points(
    model: FlowNet,
    initial_points: Tensor,
    *,
    integration_steps: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    model.to(device).eval()
    return _integrate(model, initial_points.clone().to(device), integration_steps, device)


def _integrate(model: FlowNet, points: Tensor, steps: int, device: torch.device) -> tuple[Tensor, Tensor]:
    times = torch.linspace(0, 1, steps + 1, device=device)
    trajectory = [points.detach().cpu()]
    for index in range(steps):
        points = model.step(points, times[index], times[index + 1])
        trajectory.append(points.detach().cpu())
    return torch.stack(trajectory), times.cpu()

