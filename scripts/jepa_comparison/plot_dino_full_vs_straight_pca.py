#!/usr/bin/env python
"""Plot PCA of DINOv2 Block-3 full fine-tuning vs temporal straightening latents."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.algorithms.atari100k.algorithm import _pixels_to_numpy_frames
from src.environments.environment import Environment
from src.utils.device import resolve_device
from src.utils.seeding import derive_seed, seed_everything


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_GLOB = "logs/jepa_comparison/pca_data/dino_jamesbond_straightening_pca_*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument("--straight-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/jepa_comparison/plots")
    parser.add_argument("--output-name", default="dino_full_vs_straight_pca")
    parser.add_argument("--num-states", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def latest_run_root() -> Path:
    candidates = sorted(
        (ROOT / "logs/jepa_comparison/pca_data").glob("dino_jamesbond_straightening_pca_*")
    )
    candidates = [p for p in candidates if (p / "checkpoints").exists()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run matching {DEFAULT_RUN_GLOB}")
    return candidates[-1]


def checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_root = args.run_root or latest_run_root()
    full = args.full_checkpoint or run_root / "checkpoints/full_block3/seed_1/last.pt"
    straight = (
        args.straight_checkpoint
        or run_root / "checkpoints/full_block3_straight_0p1/seed_1/last.pt"
    )
    for path in (full, straight):
        if not path.exists():
            raise FileNotFoundError(path)
    return run_root, full, straight


def compose_cfg(experiment: str, seed: int, device: torch.device):
    accelerator = "gpu" if device.type == "cuda" else device.type
    overrides = [
        f"experiment={experiment}",
        f"trainer.seed={seed}",
        f"trainer.accelerator={accelerator}",
        "trainer.devices=[0]",
        "checkpoint.enabled=false",
        "logger=[]",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train", overrides=overrides)


def build_algorithm(experiment: str, checkpoint: Path, seed: int, device: torch.device):
    cfg = compose_cfg(experiment, seed, device)
    env_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
        if k != "_target_"
    }
    environment = Environment(**env_kwargs)
    algorithm = instantiate(cfg.algorithm, device=device)
    collector_cfg = algorithm.get_collector_config()

    def make_env():
        return environment.make_env(
            num_envs=1,
            device=collector_cfg.env_device or str(device),
            seed=seed,
        )

    algorithm.setup(make_env)
    algorithm.load_checkpoint(checkpoint)
    algorithm.agent.online_network.eval()
    return algorithm, environment, cfg


def collect_stacked_states(environment: Environment, *, seed: int, num_states: int) -> np.ndarray:
    env = environment.make_env(num_envs=1, device="cpu", seed=derive_seed(seed, "pca_states"))
    frame_stack: deque[np.ndarray] = deque(maxlen=4)
    states: list[np.ndarray] = []
    try:
        td = env.reset()
        while len(states) < num_states:
            frame = _pixels_to_numpy_frames(td.get("pixels"))[0]
            if len(frame_stack) == 0:
                for _ in range(4):
                    frame_stack.append(frame)
            else:
                frame_stack.append(frame)
            states.append(np.stack(list(frame_stack), axis=-1))

            td = env.rand_step(td)
            done = (
                td["next", "done"].any().item()
                or td["next", "terminated"].any().item()
                or td["next", "truncated"].any().item()
            )
            td = env.reset() if done else td["next"]
            if done:
                frame_stack.clear()
    finally:
        env.close()
    return np.stack(states, axis=0).astype(np.uint8)


def extract_latents(algorithm, states: np.ndarray, *, batch_size: int) -> np.ndarray:
    device = algorithm.device
    chunks: list[torch.Tensor] = []
    network = algorithm.agent.online_network
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.from_numpy(states[start : start + batch_size]).to(device)
            latents = network.encode_jepa_latent(batch, eval_mode=True)
            chunks.append(latents.detach().float().cpu())
    return torch.cat(chunks, dim=0).numpy()


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


def effective_rank(latents: np.ndarray) -> dict[str, float]:
    centered = latents - latents.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values**2
    probs = singular_values / max(singular_values.sum(), 1e-12)
    entropy_rank = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))
    cumulative = np.cumsum(energy) / max(energy.sum(), 1e-12)
    return {
        "effective_rank": entropy_rank,
        "rank_95": float(np.searchsorted(cumulative, 0.95) + 1),
        "rank_99": float(np.searchsorted(cumulative, 0.99) + 1),
    }


def save_plot(projected: dict[str, np.ndarray], explained: np.ndarray, output: Path) -> None:
    colors = {"full": "#1f77b4", "straight": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, (label, points) in zip(axes, projected.items()):
        time = np.arange(points.shape[0])
        ax.plot(points[:, 0], points[:, 1], color=colors[label], alpha=0.35, linewidth=0.8)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=time,
            cmap="viridis",
            s=12,
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_title(label)
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    fig.colorbar(scatter, ax=axes, label="state index")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_points(projected: dict[str, np.ndarray], output: Path) -> None:
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "state_index", "pc1", "pc2"])
        for label, points in projected.items():
            for index, (pc1, pc2) in enumerate(points):
                writer.writerow([label, index, pc1, pc2])


def save_rank_summary(latent_sets: dict[str, np.ndarray], output: Path) -> None:
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "effective_rank", "rank_95", "rank_99"])
        for label, latents in latent_sets.items():
            stats = effective_rank(latents)
            writer.writerow([label, stats["effective_rank"], stats["rank_95"], stats["rank_99"]])


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else resolve_device("gpu" if torch.cuda.is_available() else "cpu", [0])
    run_root, full_checkpoint, straight_checkpoint = checkpoint_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_algorithm, environment, _ = build_algorithm(
        "dinov2/der/full_block3/jamesbond",
        full_checkpoint,
        args.seed,
        device,
    )
    straight_algorithm, _, _ = build_algorithm(
        "dinov2/der/jepa_full_block3/jamesbond",
        straight_checkpoint,
        args.seed,
        device,
    )

    states = collect_stacked_states(environment, seed=args.seed, num_states=args.num_states)
    latent_sets = {
        "full": extract_latents(full_algorithm, states, batch_size=args.batch_size),
        "straight": extract_latents(straight_algorithm, states, batch_size=args.batch_size),
    }
    projected, explained = pca_2d(latent_sets)

    png_path = args.output_dir / f"{args.output_name}.png"
    csv_path = args.output_dir / f"{args.output_name}.csv"
    npz_path = args.output_dir / f"{args.output_name}.npz"
    rank_path = args.output_dir / f"{args.output_name}_rank_summary.csv"

    save_plot(projected, explained, png_path)
    save_points(projected, csv_path)
    save_rank_summary(latent_sets, rank_path)
    np.savez_compressed(
        npz_path,
        states=states,
        full_latents=latent_sets["full"],
        straight_latents=latent_sets["straight"],
        full_pca=projected["full"],
        straight_pca=projected["straight"],
        explained_variance=explained,
        run_root=str(run_root),
        full_checkpoint=str(full_checkpoint),
        straight_checkpoint=str(straight_checkpoint),
    )

    print(f"Loaded run root: {run_root}")
    print(f"Saved plot: {png_path}")
    print(f"Saved points: {csv_path}")
    print(f"Saved latents: {npz_path}")
    print(f"Saved ranks: {rank_path}")


if __name__ == "__main__":
    main()
