import torch

from continualflow.datasets import load_toy_dataset


DATASETS = ("circles", "moons", "gaussians", "checkerboard")


def test_datasets_are_binary_normalized_and_deterministic():
    for name in DATASETS:
        first = load_toy_dataset(name, n_points=240, noise=0.005, seed=7)
        second = load_toy_dataset(name, n_points=240, noise=0.005, seed=7)
        assert torch.equal(first.points, second.points)
        assert torch.equal(first.retain_labels, second.retain_labels)
        assert first.points.shape == (240, 2)
        assert set(first.retain_labels.unique().tolist()) == {0.0, 1.0}
        assert float(first.points.min()) >= -0.80001
        assert float(first.points.max()) <= 0.80001


def test_dataset_retain_regions_are_explicit():
    circles = load_toy_dataset("circles", 600, 0.0, 0)
    radius = torch.linalg.vector_norm(circles.points, dim=1)
    assert radius[circles.retain_labels == 1].mean() < radius[circles.retain_labels == 0].mean()

    moons = load_toy_dataset("moons", 600, 0.0, 0)
    assert set(moons.mode_labels[moons.retain_labels == 1].unique().tolist()) == {1}

    gaussians = load_toy_dataset("gaussians", 600, 0.01, 0)
    assert set(gaussians.mode_labels[gaussians.retain_labels == 1].unique().tolist()) == {0, 2, 4}
    assert set(gaussians.mode_labels[gaussians.retain_labels == 0].unique().tolist()) == {1, 3, 5}

    checkerboard = load_toy_dataset("checkerboard", 1000, 0.0, 0)
    retained_modes = set(checkerboard.mode_labels[checkerboard.retain_labels == 1].unique().tolist())
    assert all(mode % 4 != 3 and mode // 4 != 3 for mode in retained_modes)

