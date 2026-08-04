"""Typed configuration, resumable artifacts, and CLI for the 2D experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import tempfile
import time
import tomllib
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .datasets import load_toy_dataset
from .energy import EnergyClassifier, train_energy_classifier
from .evaluation import evaluate_run
from .models import FlowNet
from .training import sample_base_flow, sample_from_points, train_base_flow, train_unlearning_flow


@dataclass(frozen=True)
class DataConfig:
    n_points: int = 5000
    noise: float = 0.005
    normalization_min: float = -0.8
    normalization_max: float = 0.8


@dataclass(frozen=True)
class ClassifierConfig:
    hidden_dim: int = 64
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-2
    subsample_size: int = 5000


@dataclass(frozen=True)
class FlowConfig:
    hidden_dim: int = 64
    base_epochs: int = 2500
    unlearning_epochs: int = 1000
    batch_size: int = 1024
    learning_rate: float = 1e-2
    suppression_lambda: float = 5.0


@dataclass(frozen=True)
class SamplingConfig:
    n_samples: int = 5000
    integration_steps: int = 10


@dataclass(frozen=True)
class PlotConfig:
    selected_indices: list[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 10])
    trajectory_resolution: int = 50
    energy_resolution: int = 300
    kde_max_points: int = 4000
    kde_bandwidth: float = 0.3
    quiver_density: int = 15
    quiver_alpha: float = 0.3
    x_min: float = -0.9
    x_max: float = 0.9
    y_min: float = -0.9
    y_max: float = 0.9
    figure_width: float = 7.0
    figure_height: float = 8.5


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    seed: int
    device: str
    datasets: list[str]
    data: DataConfig
    classifier: ClassifierConfig
    flow: FlowConfig
    sampling: SamplingConfig
    plot: PlotConfig

    def validate(self) -> None:
        expected = {"circles", "moons", "gaussians", "checkerboard"}
        if not self.datasets or not set(self.datasets).issubset(expected):
            raise ValueError(f"datasets must be a non-empty subset of {sorted(expected)}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets may not contain duplicates")
        if self.data.n_points < 2 or self.sampling.n_samples < 2:
            raise ValueError("data and sampling sizes must be at least 2")
        if self.flow.base_epochs < 1 or self.flow.unlearning_epochs < 1 or self.classifier.epochs < 1:
            raise ValueError("all epoch counts must be positive")
        maximum_index = self.sampling.integration_steps
        if any(index < 0 or index > maximum_index for index in self.plot.selected_indices):
            raise ValueError(f"plot indices must lie in [0, {maximum_index}]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RunManifest:
    run_name: str
    config_hash: str
    status: str
    seed: int
    resolved_device: str
    environment: dict[str, Any]
    datasets: dict[str, dict[str, Any]]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    config = ExperimentConfig(
        run_name=raw["run_name"],
        seed=int(raw["seed"]),
        device=raw.get("device", "auto"),
        datasets=list(raw["datasets"]),
        data=DataConfig(**raw.get("data", {})),
        classifier=ClassifierConfig(**raw.get("classifier", {})),
        flow=FlowConfig(**raw.get("flow", {})),
        sampling=SamplingConfig(**raw.get("sampling", {})),
        plot=PlotConfig(**raw.get("plot", {})),
    )
    config.validate()
    return config


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn", "tqdm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(path: Path) -> RunManifest:
    return RunManifest(**json.loads(path.read_text(encoding="utf-8")))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_metrics_csv(run_dir: Path, manifest: RunManifest) -> None:
    completed = [
        (name, state["metrics"])
        for name, state in manifest.datasets.items()
        if state.get("status") == "complete" and "metrics" in state
    ]
    if not completed:
        return
    fields = sorted({key for _, metrics in completed for key in metrics})
    path = run_dir / "metrics.csv"
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=run_dir, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", *fields])
        writer.writeheader()
        for name, metrics in completed:
            writer.writerow({"dataset": name, **metrics})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _initial_manifest(config: ExperimentConfig, device: torch.device) -> RunManifest:
    now = _timestamp()
    return RunManifest(
        run_name=config.run_name,
        config_hash=config.config_hash,
        status="pending",
        seed=config.seed,
        resolved_device=str(device),
        environment=_environment(),
        datasets={name: {"status": "pending", "seed": config.seed + index} for index, name in enumerate(config.datasets)},
        created_at=now,
        updated_at=now,
    )


def run_2d(
    config: ExperimentConfig,
    run_dir: str | Path | None = None,
    *,
    datasets: Sequence[str] | None = None,
    progress: bool = True,
) -> RunManifest:
    """Run or safely resume selected datasets without invalidating completed work."""
    config.validate()
    device = _resolve_device(config.device)
    run_dir = Path(run_dir) if run_dir is not None else Path("runs") / "2d" / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        if manifest.config_hash != config.config_hash:
            raise RuntimeError(
                f"Run directory {run_dir} belongs to config {manifest.config_hash[:12]}, "
                f"not {config.config_hash[:12]}; choose a new run_name"
            )
    else:
        unexpected = [path for path in run_dir.iterdir() if path.name != ".gitkeep"]
        if unexpected:
            raise RuntimeError(f"Refusing to use non-empty run directory without a manifest: {run_dir}")
        manifest = _initial_manifest(config, device)
        _atomic_json(manifest_path, manifest.to_dict())

    resolved = config.to_dict()
    resolved["resolved_device"] = str(device)
    _atomic_json(run_dir / "config.resolved.json", resolved)
    selected = list(datasets) if datasets is not None else list(config.datasets)
    unknown = set(selected) - set(config.datasets)
    if unknown:
        raise ValueError(f"Requested datasets are absent from the configuration: {sorted(unknown)}")
    pending = [name for name in selected if manifest.datasets[name].get("status") != "complete"]
    if not pending:
        figure = run_dir / "figures" / "toy2d_full_page.pdf"
        if manifest.status == "complete" and not figure.exists():
            from .plotting import render_full_page

            render_full_page(run_dir)
        return manifest

    manifest.status = "running"
    manifest.updated_at = _timestamp()
    _atomic_json(manifest_path, manifest.to_dict())
    for name in pending:
        dataset_dir = run_dir / "datasets" / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_seed = int(manifest.datasets[name]["seed"])
        state: dict[str, Any] = {"status": "running", "seed": dataset_seed, "started_at": _timestamp()}
        manifest.datasets[name] = state
        _atomic_json(dataset_dir / "status.json", state)
        _atomic_json(manifest_path, manifest.to_dict())
        try:
            _seed_everything(dataset_seed)
            dataset = load_toy_dataset(
                name,
                config.data.n_points,
                config.data.noise,
                dataset_seed,
                config.data.normalization_min,
                config.data.normalization_max,
            )
            _atomic_torch_save(
                dataset_dir / "dataset.pt",
                {
                    "name": dataset.name,
                    "points": dataset.points,
                    "retain_labels": dataset.retain_labels,
                    "mode_labels": dataset.mode_labels,
                    "normalization": dataset.normalization.to_dict(),
                },
            )

            subsample_count = min(config.classifier.subsample_size, len(dataset.points))
            permutation = torch.randperm(len(dataset.points), generator=torch.Generator().manual_seed(dataset_seed))
            classifier_indices = permutation[:subsample_count]
            classifier = EnergyClassifier(hidden_dim=config.classifier.hidden_dim)
            classifier_stats = train_energy_classifier(
                classifier,
                dataset.points[classifier_indices],
                dataset.retain_labels[classifier_indices],
                epochs=config.classifier.epochs,
                batch_size=config.classifier.batch_size,
                learning_rate=config.classifier.learning_rate,
                device=device,
                seed=dataset_seed,
            )

            base_model = FlowNet(hidden_dim=config.flow.hidden_dim)
            _sync(device)
            base_start = time.perf_counter()
            base_stats = train_base_flow(
                base_model,
                dataset.points,
                epochs=config.flow.base_epochs,
                batch_size=config.flow.batch_size,
                learning_rate=config.flow.learning_rate,
                device=device,
                seed=dataset_seed,
                progress=progress,
            )
            _sync(device)
            base_training_seconds = time.perf_counter() - base_start
            _sync(device)
            inference_start = time.perf_counter()
            learning_trajectory, times = sample_base_flow(
                base_model,
                n_samples=config.sampling.n_samples,
                integration_steps=config.sampling.integration_steps,
                device=device,
            )
            _sync(device)
            base_inference_seconds = time.perf_counter() - inference_start

            unlearning_model = FlowNet(hidden_dim=config.flow.hidden_dim)
            _sync(device)
            unlearning_start = time.perf_counter()
            unlearning_stats = train_unlearning_flow(
                unlearning_model,
                learning_trajectory[-1],
                classifier,
                epochs=config.flow.unlearning_epochs,
                batch_size=config.flow.batch_size,
                learning_rate=config.flow.learning_rate,
                suppression_lambda=config.flow.suppression_lambda,
                device=device,
                seed=dataset_seed,
                progress=progress,
            )
            _sync(device)
            unlearning_training_seconds = time.perf_counter() - unlearning_start
            _sync(device)
            inference_start = time.perf_counter()
            unlearning_trajectory, unlearning_times = sample_from_points(
                unlearning_model,
                learning_trajectory[-1],
                integration_steps=config.sampling.integration_steps,
                device=device,
            )
            _sync(device)
            unlearning_inference_seconds = time.perf_counter() - inference_start
            if not torch.equal(times, unlearning_times):
                raise RuntimeError("Learning and unlearning time grids differ")

            metrics = evaluate_run(
                classifier,
                unlearning_model,
                unlearning_trajectory,
                dataset.points[dataset.retain_labels == 1],
                device=device,
            )
            metrics.update(
                {
                    "base_training_seconds": base_training_seconds,
                    "unlearning_training_seconds": unlearning_training_seconds,
                    "base_inference_ms_per_sample": base_inference_seconds / config.sampling.n_samples * 1000,
                    "unlearning_inference_ms_per_sample": unlearning_inference_seconds / config.sampling.n_samples * 1000,
                }
            )
            _atomic_torch_save(dataset_dir / "classifier.pt", classifier.cpu().state_dict())
            _atomic_torch_save(dataset_dir / "base_flow.pt", base_model.cpu().state_dict())
            _atomic_torch_save(dataset_dir / "unlearning_flow.pt", unlearning_model.cpu().state_dict())
            _atomic_torch_save(
                dataset_dir / "trajectories.pt",
                {"learning": learning_trajectory, "unlearning": unlearning_trajectory, "times": times},
            )
            _atomic_json(dataset_dir / "metrics.json", metrics)
            state = {
                "status": "complete",
                "seed": dataset_seed,
                "completed_at": _timestamp(),
                "metrics": metrics,
                "optimizer_updates": {
                    "classifier": classifier_stats.optimizer_updates,
                    "base_flow": base_stats.optimizer_updates,
                    "unlearning_flow": unlearning_stats.optimizer_updates,
                },
                "final_losses": {
                    "classifier": classifier_stats.final_loss,
                    "base_flow": base_stats.final_loss,
                    "unlearning_flow": unlearning_stats.final_loss,
                },
                "artifacts": {
                    "dataset": "dataset.pt",
                    "classifier": "classifier.pt",
                    "base_flow": "base_flow.pt",
                    "unlearning_flow": "unlearning_flow.pt",
                    "trajectories": "trajectories.pt",
                    "metrics": "metrics.json",
                },
            }
        except Exception as exc:
            state = {
                "status": "failed",
                "seed": dataset_seed,
                "failed_at": _timestamp(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        manifest.datasets[name] = state
        manifest.updated_at = _timestamp()
        _atomic_json(dataset_dir / "status.json", state)
        _atomic_json(manifest_path, manifest.to_dict())

    completed = all(state.get("status") == "complete" for state in manifest.datasets.values())
    manifest.status = "complete" if completed else "partial"
    manifest.updated_at = _timestamp()
    _atomic_json(manifest_path, manifest.to_dict())
    _write_metrics_csv(run_dir, manifest)
    if completed:
        from .plotting import render_full_page

        render_full_page(run_dir)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and render ContinualFlow's reproducible 2D experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run or resume training and evaluation")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--run", type=Path, help="override the configured run directory")
    run_parser.add_argument("--dataset", action="append", choices=["circles", "moons", "gaussians", "checkerboard"])
    run_parser.add_argument("--no-progress", action="store_true")
    figure_parser = subparsers.add_parser("figure", help="regenerate the unified figure without retraining")
    figure_parser.add_argument("--run", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "figure":
        from .plotting import render_full_page

        for output in render_full_page(args.run):
            print(output)
        return 0
    config = load_config(args.config)
    manifest = run_2d(config, args.run, datasets=args.dataset, progress=not args.no_progress)
    print(json.dumps(manifest.to_dict(), indent=2))
    return 0 if manifest.status in {"complete", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
