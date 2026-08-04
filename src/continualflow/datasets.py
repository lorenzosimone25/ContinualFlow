"""Synthetic 2D datasets with one explicit retain/forget convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch import Tensor


@dataclass(frozen=True)
class Normalization:
    source_min: Tensor
    source_max: Tensor
    target_min: float = -0.8
    target_max: float = 0.8

    def transform(self, points: Tensor) -> Tensor:
        span = (self.source_max - self.source_min).clamp_min(1e-8)
        unit = (points - self.source_min) / span
        return unit * (self.target_max - self.target_min) + self.target_min

    def to_dict(self) -> dict[str, object]:
        return {
            "source_min": self.source_min.tolist(),
            "source_max": self.source_max.tolist(),
            "target_min": self.target_min,
            "target_max": self.target_max,
        }


@dataclass(frozen=True)
class ToyDataset:
    name: str
    points: Tensor
    retain_labels: Tensor
    mode_labels: Tensor
    normalization: Normalization

    def __post_init__(self) -> None:
        n = len(self.points)
        if self.points.shape != (n, 2):
            raise ValueError(f"{self.name}: expected points with shape [N, 2]")
        if self.retain_labels.shape != (n,) or self.mode_labels.shape != (n,):
            raise ValueError(f"{self.name}: labels must have shape [N]")
        unique = set(self.retain_labels.unique().tolist())
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(f"{self.name}: retain labels must be binary, got {unique}")


def _circles(n_points: int, noise: float, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    inner_radius, outer_radius = 0.3, 1.0
    n_inner = int(n_points * inner_radius / (inner_radius + outer_radius))
    n_outer = n_points - n_inner
    theta_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    theta_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner = np.column_stack((inner_radius * np.cos(theta_inner), inner_radius * np.sin(theta_inner)))
    outer = np.column_stack((outer_radius * np.cos(theta_outer), outer_radius * np.sin(theta_outer)))
    points = np.vstack((inner, outer)) + rng.normal(0.0, noise, size=(n_points, 2))
    modes = np.concatenate((np.ones(n_inner, dtype=np.int64), np.zeros(n_outer, dtype=np.int64)))
    retain = modes.copy()  # inner ring is retained
    return torch.from_numpy(points).float(), torch.from_numpy(retain).float(), torch.from_numpy(modes)


def _moons(n_points: int, noise: float, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    points, modes = make_moons(n_samples=n_points, noise=noise, random_state=seed)
    modes = modes.astype(np.int64)
    retain = (modes == 1).astype(np.float32)  # sklearn mode 1 is the lower moon
    return torch.from_numpy(points).float(), torch.from_numpy(retain), torch.from_numpy(modes)


def _gaussians(n_points: int, noise: float, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    counts = np.full(6, n_points // 6, dtype=int)
    counts[: n_points % 6] += 1
    chunks: list[np.ndarray] = []
    modes: list[np.ndarray] = []
    for mode, count in enumerate(counts):
        angle = 2 * np.pi * mode / 6
        center = 2.0 * np.array([np.cos(angle), np.sin(angle)])
        chunks.append(center + rng.normal(0.0, noise, size=(count, 2)))
        modes.append(np.full(count, mode, dtype=np.int64))
    points = np.vstack(chunks)
    mode_labels = np.concatenate(modes)
    retain = np.isin(mode_labels, [0, 2, 4]).astype(np.float32)
    return torch.from_numpy(points).float(), torch.from_numpy(retain), torch.from_numpy(mode_labels)


def _checkerboard(n_points: int, noise: float, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    n_tiles = 4
    low, high = -(n_tiles // 2) * np.pi, (n_tiles // 2) * np.pi
    accepted: list[np.ndarray] = []
    while sum(len(chunk) for chunk in accepted) < n_points:
        candidates = rng.uniform(low, high, size=(n_points, 2))
        accepted.append(candidates[np.sin(candidates[:, 0]) * np.sin(candidates[:, 1]) > 0])
    points = np.vstack(accepted)[:n_points]
    tile = np.floor((points - low) / np.pi).astype(int).clip(0, n_tiles - 1)
    mode_labels = tile[:, 1] * n_tiles + tile[:, 0]
    retain = ((tile[:, 1] != n_tiles - 1) & (tile[:, 0] != n_tiles - 1)).astype(np.float32)
    if noise:
        points = points + rng.normal(0.0, noise, size=points.shape)
    return torch.from_numpy(points).float(), torch.from_numpy(retain), torch.from_numpy(mode_labels.astype(np.int64))


_GENERATORS: dict[str, Callable[[int, float, int], tuple[Tensor, Tensor, Tensor]]] = {
    "circles": _circles,
    "moons": _moons,
    "gaussians": _gaussians,
    "checkerboard": _checkerboard,
}


def load_toy_dataset(
    name: str,
    n_points: int,
    noise: float,
    seed: int,
    normalization_min: float = -0.8,
    normalization_max: float = 0.8,
) -> ToyDataset:
    """Generate and normalize a dataset; label 1 always means retain."""
    try:
        points, retain, modes = _GENERATORS[name](n_points, noise, seed)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(_GENERATORS)}") from exc
    normalization = Normalization(
        source_min=points.amin(dim=0),
        source_max=points.amax(dim=0),
        target_min=normalization_min,
        target_max=normalization_max,
    )
    return ToyDataset(name, normalization.transform(points), retain.float(), modes.long(), normalization)

