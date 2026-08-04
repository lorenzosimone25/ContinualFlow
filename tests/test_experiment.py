import json
from dataclasses import replace

import pytest
import torch

from continualflow.experiment_2d import (
    ClassifierConfig,
    DataConfig,
    ExperimentConfig,
    FlowConfig,
    PlotConfig,
    SamplingConfig,
    run_2d,
)
from continualflow.models import FlowNet


def tiny_config(seed: int = 0) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="tiny",
        seed=seed,
        device="cpu",
        datasets=["circles", "moons", "gaussians", "checkerboard"],
        data=DataConfig(n_points=120, noise=0.01),
        classifier=ClassifierConfig(hidden_dim=8, epochs=1, batch_size=64, learning_rate=1e-2, subsample_size=120),
        flow=FlowConfig(
            hidden_dim=8,
            base_epochs=1,
            unlearning_epochs=1,
            batch_size=64,
            learning_rate=1e-2,
            suppression_lambda=5.0,
        ),
        sampling=SamplingConfig(n_samples=120, integration_steps=5),
        plot=PlotConfig(
            selected_indices=[0, 1, 2, 3, 4, 5],
            trajectory_resolution=12,
            energy_resolution=16,
            kde_max_points=100,
            kde_bandwidth=0.3,
            quiver_density=4,
            figure_width=7.0,
            figure_height=8.5,
        ),
    )


def test_tiny_all_dataset_run_resume_artifacts_and_visual_contract(tmp_path):
    config = tiny_config()
    run_dir = tmp_path / "tiny"
    manifest = run_2d(config, run_dir, progress=False)
    assert manifest.status == "complete"
    assert all(state["status"] == "complete" for state in manifest.datasets.values())
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "figures/toy2d_full_page.pdf").stat().st_size > 0
    assert (run_dir / "figures/toy2d_full_page.svg").stat().st_size > 0

    metadata = json.loads((run_dir / "figures/toy2d_full_page.metadata.json").read_text())
    assert metadata["panel_count"] == 52
    assert metadata["energy_panels"] == 4
    assert metadata["trajectory_panels"] == 48
    assert metadata["dataset_order"] == ["circles", "moons", "gaussians", "checkerboard"]
    assert metadata["palette"] == {"density": "rocket", "energy": "Blues", "vectors": "white"}

    checkpoint = torch.load(run_dir / "datasets/circles/base_flow.pt", weights_only=True)
    restored = FlowNet(hidden_dim=8)
    restored.load_state_dict(checkpoint)

    figure_mtime = (run_dir / "figures/toy2d_full_page.pdf").stat().st_mtime_ns
    resumed = run_2d(config, run_dir, progress=False)
    assert resumed.status == "complete"
    assert (run_dir / "figures/toy2d_full_page.pdf").stat().st_mtime_ns == figure_mtime


def test_config_hash_rejects_mismatched_run(tmp_path):
    config = tiny_config()
    run_dir = tmp_path / "tiny"
    run_2d(config, run_dir, datasets=["circles"], progress=False)
    changed = replace(config, seed=1)
    assert changed.config_hash != config.config_hash
    with pytest.raises(RuntimeError, match="belongs to config"):
        run_2d(changed, run_dir, datasets=["circles"], progress=False)
