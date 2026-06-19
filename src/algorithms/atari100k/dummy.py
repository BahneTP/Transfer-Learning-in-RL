"""Minimal Atari algorithm that always selects action zero."""
from __future__ import annotations

from typing import Callable

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState


class ConstantZeroPolicy(nn.Module):
    """Write action zero using the environment's categorical encoding."""

    def __init__(
        self,
        *,
        num_actions: int,
        one_hot_actions: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.one_hot_actions = one_hot_actions
        self.action_dtype = dtype
        self.action_device = device

    def forward(self, tensordict: TensorDict) -> TensorDict:
        batch_shape = tuple(tensordict.batch_size)
        device = tensordict.device or self.action_device
        if self.one_hot_actions:
            action = torch.zeros(
                (*batch_shape, self.num_actions),
                dtype=self.action_dtype,
                device=device,
            )
            action[..., 0] = 1
        else:
            action = torch.zeros(
                batch_shape,
                dtype=self.action_dtype,
                device=device,
            )
        return tensordict.set("action", action)


class DummyAtari100KAlgorithm(BaseAlgorithm):
    """No-learning baseline used to smoke-test Atari runner scripts."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        seed: int = 1,
        frames_per_batch: int = 1,
        max_frames_per_traj: int = -1,
    ) -> None:
        super().__init__(device)
        self.seed = seed
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self._processed_frames = 0
        self._policy: ConstantZeroPolicy | None = None

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        try:
            action_spec = proof_env.action_spec
            num_actions = int(action_spec.space.n)
            one_hot_actions = bool(
                len(action_spec.shape) > 0
                and action_spec.shape[-1] == num_actions
            )
            self._policy = ConstantZeroPolicy(
                num_actions=num_actions,
                one_hot_actions=one_hot_actions,
                dtype=action_spec.dtype,
                device=action_spec.device,
            )
        finally:
            proof_env.close()

    def get_policy(self) -> ConstantZeroPolicy:
        return self._require_policy()

    def get_explore_policy(self) -> ConstantZeroPolicy:
        return self._require_policy()

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            max_frames_per_traj=self.max_frames_per_traj,
            env_device="cpu",
            policy_device="cpu",
            storing_device="cpu",
        )

    def step(self, batch: TensorDict) -> dict[str, float]:
        self._processed_frames += batch.numel()
        return {
            "train/dummy_action": 0.0,
            "train/processed_frames": float(self._processed_frames),
        }

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=self._processed_frames,
            policy_state_dict={},
            optimizer_state_dict={},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self._processed_frames = int(state.step)

    def _require_policy(self) -> ConstantZeroPolicy:
        if self._policy is None:
            raise RuntimeError("DummyAtari100KAlgorithm.setup() must be called first.")
        return self._policy
