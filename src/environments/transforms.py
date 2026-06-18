"""Repository-specific TorchRL environment transforms."""
from __future__ import annotations

import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite
from torchrl.envs.transforms import Transform


class MaxAndSkipTransform(Transform):
    """Repeat an action and max-pool the last two raw pixel observations.

    This mirrors the Atari wrapper used by ``BBF-pytorch``:
    repeat the selected action ``frame_skip`` times, sum rewards, stop early
    on episode end, and return ``max(obs[-2], obs[-1])`` for the pixel key.
    Place this before image preprocessing transforms.
    """

    def __init__(self, frame_skip: int = 4, pixel_key: str = "pixels") -> None:
        super().__init__()
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.frame_skip = frame_skip
        self.pixel_key = pixel_key

    def _step(
        self,
        tensordict: TensorDictBase,
        next_tensordict: TensorDictBase,
    ) -> TensorDictBase:
        parent = self.parent
        if parent is None:
            raise RuntimeError("parent not found for MaxAndSkipTransform")

        reward_key = parent.reward_key
        reward = next_tensordict.get(reward_key)
        pixel_buffer = [next_tensordict.get(self.pixel_key, default=None)]

        for _ in range(self.frame_skip - 1):
            if _done(next_tensordict):
                break
            next_tensordict = parent._step(tensordict)
            reward = reward + next_tensordict.get(reward_key)
            pixel_buffer.append(next_tensordict.get(self.pixel_key, default=None))
            pixel_buffer = pixel_buffer[-2:]

        valid_pixels = [pixels for pixels in pixel_buffer if pixels is not None]
        if len(valid_pixels) == 2:
            next_tensordict.set(
                self.pixel_key,
                torch.maximum(valid_pixels[0], valid_pixels[1]),
            )
        return next_tensordict.set(reward_key, reward)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(
            "MaxAndSkipTransform can only be used when appended to a transformed env."
        )


def _done(tensordict: TensorDictBase) -> bool:
    for key in ("done", "terminated", "truncated"):
        value = tensordict.get(key, default=None)
        if value is not None and bool(value.any().item()):
            return True
    return False


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
