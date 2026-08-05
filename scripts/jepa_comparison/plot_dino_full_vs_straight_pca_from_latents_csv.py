#!/usr/bin/env python
"""Plot PCA from a CSV containing DINOv2 full/straight latents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-csv", type=Path, default=ROOT / "logs/jepa_comparison/pca_data/dino_full_vs_straight_latents.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/jepa_comparison/plots")
    parser.add_argument("--output-name", default="dino_full_vs_straight_pca")
    parser.add_argument("--max-states", type=int, default=300)
    return parser.parse_args()


def load_latents(
    path: Path,
    *,
    max_states: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    grouped: dict[str, list[list[float]]] = {}
    episodes: dict[str, list[int]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            variant = row["variant"]
            state_index = int(row["state_index"])
            if max_states is not None and state_index >= max_states:
                continue
            episode_index = int(row.get("episode_index", row["state_index"]))
            latent = [
                float(value)
                for key, value in row.items()
                if key.startswith("latent_")
            ]
            grouped.setdefault(variant, []).append(latent)
            episodes.setdefault(variant, []).append(episode_index)
    if not grouped:
        raise ValueError(f"No latents found in {path}")
    return (
        {variant: np.asarray(values, dtype=np.float32) for variant, values in grouped.items()},
        {variant: np.asarray(values, dtype=np.int32) for variant, values in episodes.items()},
    )


def pca_2d(latent_sets: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    labels = list(latent_sets)
    combined = np.concatenate([latent_sets[label] for label in labels], axis=0)
    mean = combined.mean(axis=0, keepdims=True)
    centered = combined - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    projected = {
        label: (values - mean) @ components
        for label, values in latent_sets.items()
    }
    explained = np.var(centered @ components, axis=0) / np.var(centered, axis=0).sum()
    return projected, explained


def save_plot(
    projected: dict[str, np.ndarray],
    episode_indices: dict[str, np.ndarray],
    explained: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (label, points) in zip(axes, projected.items()):
        draw_trajectory(ax, label, points, episode_indices[label], explained)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def draw_trajectory(
    ax,
    label: str,
    points: np.ndarray,
    episode_ids: np.ndarray,
    explained: np.ndarray,
) -> None:
    cmap = plt.get_cmap("tab10")
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        color = cmap(int(episode_id) % 10)
        episode_points = points[mask]
        ax.plot(
            episode_points[:, 0],
            episode_points[:, 1],
            color=color,
            alpha=0.35,
            linewidth=0.8,
        )
        ax.scatter(
            episode_points[:, 0],
            episode_points[:, 1],
            color=color,
            s=12,
            alpha=0.8,
            edgecolors="none",
            label=f"episode {int(episode_id)}",
        )
    ax.set_title(label)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(alpha=0.2)
    ax.margins(0.08)
    ax.legend(loc="best", fontsize=8)


def save_individual_plots(
    projected: dict[str, np.ndarray],
    episode_indices: dict[str, np.ndarray],
    explained: np.ndarray,
    output_dir: Path,
    output_name: str,
) -> list[Path]:
    paths = []
    for label, points in projected.items():
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        draw_trajectory(ax, label, points, episode_indices[label], explained)
        output = output_dir / f"{output_name}_{label}.png"
        fig.savefig(output, dpi=200)
        plt.close(fig)
        paths.append(output)
    return paths


def save_points(
    projected: dict[str, np.ndarray],
    episode_indices: dict[str, np.ndarray],
    output: Path,
) -> None:
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "state_index", "episode_index", "pc1", "pc2"])
        for label, points in projected.items():
            for index, (pc1, pc2) in enumerate(points):
                writer.writerow([label, index, int(episode_indices[label][index]), pc1, pc2])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_sets, episode_indices = load_latents(args.latents_csv, max_states=args.max_states)
    projected, explained = pca_2d(latent_sets)

    png_path = args.output_dir / f"{args.output_name}.png"
    points_path = args.output_dir / f"{args.output_name}.csv"
    save_plot(projected, episode_indices, explained, png_path)
    individual_paths = save_individual_plots(
        projected,
        episode_indices,
        explained,
        args.output_dir,
        args.output_name,
    )
    save_points(projected, episode_indices, points_path)

    print(f"Loaded latents CSV: {args.latents_csv}")
    print(f"Saved plot: {png_path}")
    for path in individual_paths:
        print(f"Saved individual plot: {path}")
    print(f"Saved points: {points_path}")


if __name__ == "__main__":
    main()
