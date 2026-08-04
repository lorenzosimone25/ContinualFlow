import torch

from continualflow.datasets import load_toy_dataset
from continualflow.energy import EnergyClassifier, forget_energy, retention_weights, train_energy_classifier
from continualflow.models import FlowNet
from continualflow.training import sample_base_flow, train_unlearning_flow


def test_energy_is_high_and_weight_is_low_for_forget_predictions():
    classifier = EnergyClassifier(input_dim=2, hidden_dim=1)
    with torch.no_grad():
        classifier.net[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        classifier.net[0].bias.zero_()
        classifier.net[2].weight.fill_(1.0)
        classifier.net[2].bias.zero_()
        classifier.net[4].weight.fill_(1.0)
        classifier.net[4].bias.fill_(-1.0)
    retain_point = torch.tensor([[3.0, 0.0]])
    forget_point = torch.tensor([[-3.0, 0.0]])
    assert forget_energy(classifier, forget_point) > forget_energy(classifier, retain_point)
    assert retention_weights(classifier, forget_point, 5.0) < retention_weights(classifier, retain_point, 5.0)


def test_trained_energy_orientation_for_every_dataset():
    for index, name in enumerate(("circles", "moons", "gaussians", "checkerboard")):
        seed = 3 + index
        dataset = load_toy_dataset(name, 600, 0.005, seed)
        torch.manual_seed(seed)
        classifier = EnergyClassifier(hidden_dim=32)
        train_energy_classifier(
            classifier,
            dataset.points,
            dataset.retain_labels,
            epochs=30,
            batch_size=128,
            learning_rate=1e-2,
            device=torch.device("cpu"),
            seed=seed,
        )
        with torch.no_grad():
            energy = forget_energy(classifier, dataset.points)
            weights = retention_weights(classifier, dataset.points, 5.0)
        assert energy[dataset.retain_labels == 0].mean() > energy[dataset.retain_labels == 1].mean()
        assert weights[dataset.retain_labels == 0].mean() < weights[dataset.retain_labels == 1].mean()


def test_model_shapes_sampling_and_finite_weighted_loss():
    device = torch.device("cpu")
    model = FlowNet(hidden_dim=8)
    points = torch.randn(16, 2)
    times = torch.rand(16, 1)
    assert model(points, times).shape == points.shape
    trajectory, time_grid = sample_base_flow(model, n_samples=16, integration_steps=3, device=device)
    assert trajectory.shape == (4, 16, 2)
    assert torch.allclose(time_grid, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))

    classifier = EnergyClassifier(hidden_dim=8)
    stats = train_unlearning_flow(
        model,
        trajectory[-1],
        classifier,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        suppression_lambda=5.0,
        device=device,
        seed=0,
        progress=False,
    )
    assert stats.optimizer_updates == 2
    assert torch.isfinite(torch.tensor(stats.final_loss))
