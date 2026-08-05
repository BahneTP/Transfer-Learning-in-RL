#!/usr/bin/env python
"""Collect Jamesbond states and save DINOv2 full/straight/SIGReg latents as CSV."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

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
DEFAULT_SIGREG_RUN_GLOB = "logs/jepa_comparison/pca_data/dino_jamesbond_full_block3_sigreg_0p1_checkpoint_*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--sigreg-run-root", type=Path, default=None)
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument("--straight-checkpoint", type=Path, default=None)
    parser.add_argument("--sigreg-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/jepa_comparison/pca_data/dino_full_vs_straight_latents.csv")
    parser.add_argument("--per-model-output-dir", type=Path, default=ROOT / "logs/jepa_comparison/pca_data")
    parser.add_argument("--num-states", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def latest_run_root() -> Path:
    candidates = sorted((ROOT / "logs/jepa_comparison/pca_data").glob("dino_jamesbond_straightening_pca_*"))
    candidates = [p for p in candidates if (p / "checkpoints").exists()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run matching {DEFAULT_RUN_GLOB}")
    return candidates[-1]


def latest_sigreg_run_root() -> Path:
    candidates = sorted((ROOT / "logs/jepa_comparison/pca_data").glob("dino_jamesbond_full_block3_sigreg_0p1_checkpoint_*"))
    candidates = [p for p in candidates if (p / "checkpoints").exists()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run matching {DEFAULT_SIGREG_RUN_GLOB}")
    return candidates[-1]


def checkpoint_paths(args: argparse.Namespace) -> tuple[Path, dict[str, Path]]:
    run_root = args.run_root or latest_run_root()
    sigreg_run_root = args.sigreg_run_root or latest_sigreg_run_root()
    full = args.full_checkpoint or run_root / "checkpoints/full_block3/seed_1/last.pt"
    straight = args.straight_checkpoint or run_root / "checkpoints/full_block3_straight_0p1/seed_1/last.pt"
    sigreg = args.sigreg_checkpoint or sigreg_run_root / "checkpoints/full_block3_sigreg_0p1/seed_1/last.pt"
    checkpoints = {
        "full": full,
        "straight": straight,
        "sigreg": sigreg,
    }
    for path in checkpoints.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return run_root, checkpoints


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
    return algorithm, environment


def collect_stacked_states(
    environment: Environment,
    *,
    seed: int,
    num_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    env = environment.make_env(num_envs=1, device="cpu", seed=derive_seed(seed, "pca_states"))
    frame_stack: deque[np.ndarray] = deque(maxlen=4)
    states: list[np.ndarray] = []
    episode_indices: list[int] = []
    episode_index = 0
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
            episode_indices.append(episode_index)

            td = env.rand_step(td)
            done = (
                td["next", "done"].any().item()
                or td["next", "terminated"].any().item()
                or td["next", "truncated"].any().item()
            )
            td = env.reset() if done else td["next"]
            if done:
                frame_stack.clear()
                episode_index += 1
    finally:
        env.close()
    return np.stack(states, axis=0).astype(np.uint8), np.asarray(episode_indices, dtype=np.int32)


def extract_latents(algorithm, states: np.ndarray, *, batch_size: int) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    network = algorithm.agent.online_network
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.from_numpy(states[start : start + batch_size]).to(algorithm.device)
            latents = network.encode_jepa_latent(batch, eval_mode=True)
            chunks.append(latents.detach().float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def save_latents_csv(
    path: Path,
    latent_sets: dict[str, np.ndarray],
    episode_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latent_dim = next(iter(latent_sets.values())).shape[1]
    header = ["variant", "state_index", "episode_index", *[f"latent_{i}" for i in range(latent_dim)]]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for variant, latents in latent_sets.items():
            for index, latent in enumerate(latents):
                writer.writerow([variant, index, int(episode_indices[index]), *latent.tolist()])


def save_per_model_latents_csv(
    output_dir: Path,
    latent_sets: dict[str, np.ndarray],
    episode_indices: np.ndarray,
) -> list[Path]:
    paths = []
    for variant, latents in latent_sets.items():
        output = output_dir / f"dino_{variant}_latents.csv"
        save_latents_csv(output, {variant: latents}, episode_indices)
        paths.append(output)
    return paths


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else resolve_device("gpu" if torch.cuda.is_available() else "cpu", [0])
    run_root, checkpoints = checkpoint_paths(args)

    full_algorithm, environment = build_algorithm(
        "dinov2/der/full_block3/jamesbond",
        checkpoints["full"],
        args.seed,
        device,
    )
    straight_algorithm, _ = build_algorithm(
        "dinov2/der/jepa_full_block3/jamesbond",
        checkpoints["straight"],
        args.seed,
        device,
    )
    sigreg_algorithm, _ = build_algorithm(
        "dinov2/der/jepa_full_block3/jamesbond",
        checkpoints["sigreg"],
        args.seed,
        device,
    )

    states, episode_indices = collect_stacked_states(environment, seed=args.seed, num_states=args.num_states)
    latent_sets = {
        "full": extract_latents(full_algorithm, states, batch_size=args.batch_size),
        "straight": extract_latents(straight_algorithm, states, batch_size=args.batch_size),
        "sigreg": extract_latents(sigreg_algorithm, states, batch_size=args.batch_size),
    }
    save_latents_csv(
        args.output,
        {
            "full": latent_sets["full"],
            "straight": latent_sets["straight"],
        },
        episode_indices,
    )
    per_model_paths = save_per_model_latents_csv(
        args.per_model_output_dir,
        latent_sets,
        episode_indices,
    )

    print(f"Loaded run root: {run_root}")
    print(f"Collected states: {len(states)}")
    print(f"Collected episodes: {int(episode_indices.max()) + 1 if len(episode_indices) else 0}")
    print(f"Saved combined latents CSV: {args.output}")
    for path in per_model_paths:
        print(f"Saved model latents CSV: {path}")


if __name__ == "__main__":
    main()
