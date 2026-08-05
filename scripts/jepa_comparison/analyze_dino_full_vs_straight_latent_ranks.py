#!/usr/bin/env python
"""Compute effective latent ranks from DINOv2 latent CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latents-csvs",
        nargs="*",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs/jepa_comparison/plots/dino_latent_rank_summary.csv",
    )
    parser.add_argument("--max-states", type=int, default=None)
    return parser.parse_args()


def default_latent_csvs() -> list[Path]:
    paths = [
        ROOT / "logs/jepa_comparison/pca_data/dino_full_latents.csv",
        ROOT / "logs/jepa_comparison/pca_data/dino_straight_latents.csv",
        ROOT / "logs/jepa_comparison/pca_data/dino_sigreg_latents.csv",
    ]
    if all(path.exists() for path in paths):
        return paths
    return [ROOT / "logs/jepa_comparison/pca_data/dino_full_vs_straight_latents.csv"]


def load_latents(path: Path, *, max_states: int | None) -> dict[str, np.ndarray]:
    grouped: dict[str, list[list[float]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            state_index = int(row["state_index"])
            if max_states is not None and state_index >= max_states:
                continue
            latent = [
                float(value)
                for key, value in row.items()
                if key.startswith("latent_")
            ]
            grouped.setdefault(row["variant"], []).append(latent)
    if not grouped:
        raise ValueError(f"No latents found in {path}")
    return {variant: np.asarray(values, dtype=np.float32) for variant, values in grouped.items()}


def load_all_latents(paths: list[Path], *, max_states: int | None) -> dict[str, np.ndarray]:
    latent_sets: dict[str, np.ndarray] = {}
    for path in paths:
        for variant, latents in load_latents(path, max_states=max_states).items():
            latent_sets[variant] = latents
    return latent_sets


def rank_stats(latents: np.ndarray) -> dict[str, float]:
    centered = latents - latents.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values**2
    singular_sum = max(float(singular_values.sum()), 1e-12)
    energy_sum = max(float(energy.sum()), 1e-12)
    probabilities = singular_values / singular_sum
    cumulative_energy = np.cumsum(energy) / energy_sum
    return {
        "num_states": float(latents.shape[0]),
        "latent_dim": float(latents.shape[1]),
        "effective_rank": float(np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum())),
        "rank_90": float(np.searchsorted(cumulative_energy, 0.90) + 1),
        "rank_95": float(np.searchsorted(cumulative_energy, 0.95) + 1),
        "rank_99": float(np.searchsorted(cumulative_energy, 0.99) + 1),
    }


def main() -> None:
    args = parse_args()
    paths = args.latents_csvs or default_latent_csvs()
    latent_sets = load_all_latents(paths, max_states=args.max_states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, latents in latent_sets.items():
        stats = rank_stats(latents)
        rows.append({"variant": variant, **stats})

    with args.output.open("w", newline="") as f:
        fieldnames = ["variant", "num_states", "latent_dim", "effective_rank", "rank_90", "rank_95", "rank_99"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for path in paths:
        print(f"Loaded latents CSV: {path}")
    for row in rows:
        print(
            f"{row['variant']}: n={int(row['num_states'])}, "
            f"effective_rank={row['effective_rank']:.2f}, "
            f"rank_95={int(row['rank_95'])}, rank_99={int(row['rank_99'])}"
        )
    print(f"Saved ranks: {args.output}")


if __name__ == "__main__":
    main()
