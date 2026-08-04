"""Axis-oriented plotting and the unified four-dataset manuscript figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

from .energy import EnergyClassifier, forget_energy
from .models import FlowNet


FIGURE_CLAIM = (
    "Energy reweighting redirects each learned flow away from its forget region "
    "while preserving retained structure across four geometries."
)
DATASET_TITLES = {
    "circles": "Circles",
    "moons": "Moons",
    "gaussians": "6 Gaussians",
    "checkerboard": "Checkerboard",
}


def truncate_colormap(cmap: Any, minimum: float = 0.1, maximum: float = 0.9, n: int = 256) -> LinearSegmentedColormap:
    colors = cmap(np.linspace(minimum, maximum, n))
    return LinearSegmentedColormap.from_list("truncated", colors)


def render_energy_axis(
    ax: Axes,
    points: torch.Tensor,
    classifier: EnergyClassifier,
    *,
    device: torch.device,
    resolution: int,
) -> None:
    """Render the existing density-over-energy visual into a supplied 3D axis."""
    blues = truncate_colormap(sns.color_palette("Blues", as_cmap=True), 0.2, 0.8)
    rocket = truncate_colormap(sns.color_palette("rocket", as_cmap=True), 0.2, 0.8)
    point_array = points.detach().cpu().numpy()
    kde = gaussian_kde(point_array.T, bw_method=0.3)
    limits = (-1.5, 1.5)
    grid_x, grid_y = np.meshgrid(np.linspace(*limits, resolution), np.linspace(*limits, resolution))
    coordinates = np.vstack((grid_x.ravel(), grid_y.ravel()))
    density = kde(coordinates).reshape(grid_x.shape)
    density = density / max(float(density.max()), np.finfo(float).eps)
    classifier.to(device).eval()
    query = torch.tensor(coordinates.T, dtype=torch.float32, device=device)
    with torch.no_grad():
        energy = forget_energy(classifier, query).cpu().numpy().reshape(grid_x.shape)
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8) * 0.5
    ax.plot_surface(grid_x, grid_y, density + 1.82, cmap=rocket, edgecolor="none", antialiased=True)
    ax.contourf(grid_x, grid_y, energy, zdir="z", offset=0, levels=10, cmap=blues, alpha=1)
    ax.plot_surface(grid_x, grid_y, np.zeros_like(grid_x), alpha=0.03, color="gray", edgecolor="none")
    ax.set(xlim=limits, ylim=limits, zlim=(0, 2), xlabel="$x_1$", ylabel="$x_2$")
    ax.set_zticks([])
    ax.tick_params(labelsize=4, pad=0)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    ax.set_facecolor("white")
    ax.view_init(elev=30, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))


def render_trajectory_axis(
    ax: Axes,
    points: torch.Tensor,
    time: float,
    model: FlowNet,
    *,
    device: torch.device,
    resolution: int,
    max_kde_points: int,
    bandwidth: float,
    quiver_density: int,
    quiver_alpha: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    rng: np.random.Generator,
) -> None:
    point_array = points.detach().cpu().numpy()
    if len(point_array) > max_kde_points:
        point_array = point_array[rng.choice(len(point_array), max_kde_points, replace=False)]
    grid_x, grid_y = np.meshgrid(np.linspace(*xlim, resolution), np.linspace(*ylim, resolution))
    grid_coordinates = np.vstack((grid_x.ravel(), grid_y.ravel()))
    density = gaussian_kde(point_array.T, bw_method=bandwidth)(grid_coordinates).reshape(resolution, resolution)
    ax.imshow(density, origin="lower", extent=(*xlim, *ylim), cmap="rocket", aspect="equal")

    quiver_x, quiver_y = np.meshgrid(np.linspace(*xlim, quiver_density), np.linspace(*ylim, quiver_density))
    quiver = np.column_stack((quiver_x.ravel(), quiver_y.ravel()))
    query = torch.tensor(quiver, dtype=torch.float32, device=device)
    times = torch.full((len(query), 1), time, device=device)
    model.to(device).eval()
    with torch.no_grad():
        velocity = model(query, times).cpu().numpy()
    ax.quiver(
        quiver[:, 0],
        quiver[:, 1],
        velocity[:, 0],
        velocity[:, 1],
        color="white",
        alpha=quiver_alpha,
        scale=10,
        width=0.006,
        headwidth=8,
        headlength=10,
        headaxislength=4,
    )
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_paths(run_dir: Path, datasets: Iterable[str]) -> list[tuple[str, Path]]:
    paths = [(name, run_dir / "datasets" / name) for name in datasets]
    missing = [str(path) for _, path in paths if not (path / "status.json").exists()]
    if missing:
        raise FileNotFoundError(f"Run is missing dataset artifacts: {missing}")
    incomplete = [name for name, path in paths if _load_json(path / "status.json").get("status") != "complete"]
    if incomplete:
        raise RuntimeError(f"Cannot render incomplete datasets: {incomplete}")
    return paths


def render_full_page(run_dir: str | Path, formats: tuple[str, ...] = ("pdf", "svg")) -> list[Path]:
    """Render one vector page directly from a completed run's scientific artifacts."""
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.resolved.json")
    datasets = config["datasets"]
    selected = config["plot"]["selected_indices"]
    if len(selected) != 6:
        raise ValueError("The manuscript page requires exactly six selected trajectory times")
    device_name = config.get("resolved_device", "cpu")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    sns.set_style("white")
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(config["plot"]["figure_width"], config["plot"]["figure_height"]))
    outer = fig.add_gridspec(len(datasets), 2, width_ratios=(0.23, 0.70), hspace=0.20, wspace=0.06)
    rng = np.random.default_rng(config["seed"])
    panel_count = 0

    for row, (name, dataset_dir) in enumerate(_dataset_paths(run_dir, datasets)):
        stored = torch.load(dataset_dir / "dataset.pt", map_location="cpu", weights_only=False)
        trajectories = torch.load(dataset_dir / "trajectories.pt", map_location="cpu", weights_only=False)
        classifier = EnergyClassifier(hidden_dim=config["classifier"]["hidden_dim"])
        classifier.load_state_dict(torch.load(dataset_dir / "classifier.pt", map_location="cpu", weights_only=True))
        base_model = FlowNet(hidden_dim=config["flow"]["hidden_dim"])
        base_model.load_state_dict(torch.load(dataset_dir / "base_flow.pt", map_location="cpu", weights_only=True))
        unlearning_model = FlowNet(hidden_dim=config["flow"]["hidden_dim"])
        unlearning_model.load_state_dict(torch.load(dataset_dir / "unlearning_flow.pt", map_location="cpu", weights_only=True))

        energy_axis = fig.add_subplot(outer[row, 0], projection="3d")
        render_energy_axis(
            energy_axis,
            stored["points"],
            classifier,
            device=device,
            resolution=config["plot"]["energy_resolution"],
        )
        energy_axis.set_title(DATASET_TITLES[name], fontsize=7, pad=-2)
        panel_count += 1

        flow_grid = outer[row, 1].subgridspec(2, 6, hspace=0.06, wspace=0.04)
        for flow_row, (trajectory_key, model, row_title) in enumerate(
            (("learning", base_model, "Learning"), ("unlearning", unlearning_model, "Unlearning"))
        ):
            for column, trajectory_index in enumerate(selected):
                axis = fig.add_subplot(flow_grid[flow_row, column])
                time = float(trajectories["times"][trajectory_index])
                render_trajectory_axis(
                    axis,
                    trajectories[trajectory_key][trajectory_index],
                    time,
                    model,
                    device=device,
                    resolution=config["plot"]["trajectory_resolution"],
                    max_kde_points=config["plot"]["kde_max_points"],
                    bandwidth=config["plot"]["kde_bandwidth"],
                    quiver_density=config["plot"]["quiver_density"],
                    quiver_alpha=config["plot"]["quiver_alpha"],
                    xlim=(config["plot"]["x_min"], config["plot"]["x_max"]),
                    ylim=(config["plot"]["y_min"], config["plot"]["y_max"]),
                    rng=rng,
                )
                if flow_row == 0:
                    axis.set_title(f"t = {time:.2f}", fontsize=5, pad=1)
                if column == 0:
                    axis.set_ylabel(row_title, fontsize=6, labelpad=2)
                panel_count += 1

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.99, bottom=0.01)
    outputs: list[Path] = []
    for extension in formats:
        output = figure_dir / f"toy2d_full_page.{extension}"
        fig.savefig(output, format=extension, dpi=300, transparent=True)
        outputs.append(output)
    plt.close(fig)
    metadata = {
        "claim": FIGURE_CLAIM,
        "dataset_order": datasets,
        "selected_indices": selected,
        "selected_times": [index / config["sampling"]["integration_steps"] for index in selected],
        "panel_count": panel_count,
        "energy_panels": len(datasets),
        "trajectory_panels": len(datasets) * 2 * len(selected),
        "roles": {
            "energy": "methodological bridge",
            "learning": "original transport",
            "unlearning": "claim-supporting evidence",
        },
        "palette": {"density": "rocket", "energy": "Blues", "vectors": "white"},
    }
    (figure_dir / "toy2d_full_page.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return outputs
