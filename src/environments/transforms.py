"""Repository-specific environment transforms."""
from __future__ import annotations

from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite
from torchrl.envs.transforms import Transform


class RewardSnapshotTransform(Transform):
    """Copy a reward tensor to another key before later transforms modify it."""

    def __init__(self, in_key: str = "reward", out_key: str = "raw_reward"):
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.in_key = in_key
        self.out_key = out_key

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        reward = next_tensordict.get(self.in_key)
        return next_tensordict.set(self.out_key, reward.clone())

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        spec = reward_spec.clone()
        spec.set(self.out_key, reward_spec[self.in_key].clone())
        return spec
