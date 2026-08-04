"""Binary retain classifier and explicitly forget-oriented energy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class EnergyClassifier(nn.Module):
    """Returns a logit for P(retain | x)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class ClassifierTrainingStats:
    final_loss: float
    optimizer_updates: int


def train_energy_classifier(
    model: EnergyClassifier,
    points: Tensor,
    retain_labels: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> ClassifierTrainingStats:
    model.to(device).train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(points.cpu(), retain_labels.float().cpu()),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    updates, final_loss = 0, float("nan")
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = F.binary_cross_entropy_with_logits(model(x_batch), y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            final_loss = float(loss.detach().cpu())
    model.eval()
    return ClassifierTrainingStats(final_loss, updates)


def forget_energy(classifier: EnergyClassifier, points: Tensor) -> Tensor:
    """Return -log P(retain|x), hence high energy on predicted forget samples."""
    return F.softplus(-classifier(points))


def retention_weights(classifier: EnergyClassifier, points: Tensor, suppression_lambda: float) -> Tensor:
    energy = forget_energy(classifier, points)
    weights = torch.sigmoid(-suppression_lambda * energy)
    if not torch.isfinite(weights).all():
        raise FloatingPointError("Energy weighting produced non-finite values")
    return weights

