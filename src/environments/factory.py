"""Environment factory for gymnasium-backed TorchRL envs.

Builds a (possibly vectorised) ``TransformedEnv`` from a small parameter
set and an explicit list of transform descriptors.

Each transform descriptor is a dict with a ``_target_`` key (a dotted path
to a ``torchrl.envs.transforms`` class) plus its constructor kwargs.
Transforms are instantiated fresh per ``make_env()`` call so each env has
independent transform state.
"""
from __future__ import annotations

import importlib
from contextlib import nullcontext
from functools import partial
from typing import Sequence


def make_env(
    name: str,
    num_envs: int = 1,
    device: str = "cpu",
    transforms: list | None = None,
    gym_kwargs: dict | None = None,
    gym_backend: str | None = None,
    atari_preprocessing: dict | None = None,
    seed: int | None = None,
    **_: object,
):
    """Build a (possibly vectorised) ``TransformedEnv`` for a gymnasium env.

    Args:
        name: gymnasium env name (e.g. ``"CartPole-v1"``).
        num_envs: number of parallel envs (>1 -> ``ParallelEnv``).
        device: target device string. ``ParallelEnv`` workers always run on
            CPU because CUDA contexts cannot survive ``fork``; the collector
            moves data to ``device`` after collection.
        seed: optional base seed. Batched environments deterministically derive
            a distinct seed for each worker.
        transforms: list of ``_target_``-keyed dicts to apply on top of the
            base env. ``None`` or empty -> bare base env.
        gym_kwargs: extra kwargs passed straight to ``GymEnv`` (e.g.
            ``{"frame_skip": 4, "from_pixels": True}``).
        gym_backend: optional gym backend name for ``set_gym_backend``
            (e.g. ``"gymnasium"``); if ``None`` torchrl picks the default.
    """
    worker_device = "cpu" if num_envs > 1 else device

    env_fn = partial(
        _make_gymnasium_env,
        name=name,
        transforms=transforms,
        device=worker_device,
        gym_kwargs=gym_kwargs,
        gym_backend=gym_backend,
        atari_preprocessing=atari_preprocessing,
    )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv

        env = ParallelEnv(num_envs, env_fn, mp_start_method="spawn")
    else:
        env = env_fn()

    if seed is not None:
        # Batched envs deterministically derive a distinct seed per worker.
        env.set_seed(seed)
    return env


def _instantiate_transform(cfg: dict):
    """Instantiate a transform from a ``_target_``-keyed dict (no Hydra runtime)."""
    cfg = dict(cfg)  # copy — don't mutate the caller
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**cfg)


def _make_gymnasium_env(
    name: str,
    transforms: list | None,
    device: str,
    gym_kwargs: dict | None = None,
    gym_backend: str | None = None,
    atari_preprocessing: dict | None = None,
):
    from torchrl.envs import GymEnv, GymWrapper, TransformedEnv
    from torchrl.envs.transforms import Compose

    backend_ctx = nullcontext()
    if gym_backend is not None:
        from torchrl.envs import set_gym_backend
        backend_ctx = set_gym_backend(gym_backend)

    with backend_ctx:
        if atari_preprocessing is None:
            base_env = GymEnv(name, device=device, **(gym_kwargs or {}))
        else:
            import ale_py
            import gymnasium as gym

            from src.environments.atari_wrappers import wrap_atari

            if hasattr(gym, "register_envs"):
                gym.register_envs(ale_py)
            kwargs = dict(gym_kwargs or {})
            from_pixels = bool(kwargs.pop("from_pixels", False))
            pixels_only = bool(kwargs.pop("pixels_only", False))
            categorical = bool(kwargs.pop("categorical_action_encoding", False))
            if from_pixels:
                kwargs.setdefault("render_mode", "rgb_array")
            raw_env = gym.make(name, **kwargs)
            raw_env = wrap_atari(raw_env, **dict(atari_preprocessing))
            base_env = GymWrapper(
                raw_env,
                device=device,
                from_pixels=from_pixels,
                pixels_only=pixels_only,
                categorical_action_encoding=categorical,
            )

    if not transforms:
        return base_env

    transform_objects = [_instantiate_transform(t) for t in transforms]
    return TransformedEnv(base_env, Compose(*transform_objects))
