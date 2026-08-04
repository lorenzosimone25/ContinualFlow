"""Neural models used by the 2D experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FlowNet(nn.Module):
    def __init__(self, dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x), dim=-1))

    def step(self, x: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = _expand_time(t_start, len(x))
        t_end = _expand_time(t_end, len(x))
        delta = t_end - t_start
        midpoint_t = t_start + delta / 2
        midpoint_x = x + self(x, t_start) * (delta / 2)
        return x + delta * self(midpoint_x, midpoint_t)


def _expand_time(t: Tensor, batch_size: int) -> Tensor:
    if t.ndim == 0:
        return t.reshape(1, 1).expand(batch_size, 1)
    if t.ndim == 1:
        return t.reshape(-1, 1).expand(batch_size, 1)
    return t

