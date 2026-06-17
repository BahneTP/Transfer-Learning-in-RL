"""Repository-specific TorchRL environment transforms."""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDictBase
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


class OpenCVResizeTransform(Transform):
    """Resize image observations with OpenCV's area interpolation.

    BBF-pytorch preprocesses Atari frames with ``cv2.resize(..., INTER_AREA)``.
    TorchRL's ``Resize`` defaults to bilinear interpolation, which produces
    different pixels after downsampling.
    """

    def __init__(
        self,
        w: int,
        h: int | None = None,
        *,
        in_keys: list[str] | None = None,
        out_keys: list[str] | None = None,
    ) -> None:
        resolved_in_keys = in_keys or ["pixels"]
        super().__init__(in_keys=resolved_in_keys, out_keys=out_keys or resolved_in_keys)
        self.w = int(w)
        self.h = int(h if h is not None else w)

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        device = obs.device
        leading_shape = obs.shape[:-3] if obs.ndim > 3 else ()
        flat = obs.reshape((-1, *obs.shape[-3:])) if leading_shape else obs.unsqueeze(0)
        resized = [_resize_one(frame, self.w, self.h) for frame in flat]
        out = torch.stack(resized, dim=0)
        if leading_shape:
            out = out.reshape((*leading_shape, *out.shape[-3:]))
        else:
            out = out[0]
        return out.to(device=device)

    def _reset(
        self,
        tensordict: TensorDictBase,
        tensordict_reset: TensorDictBase,
    ) -> TensorDictBase:
        return self._call(tensordict_reset)


def _resize_one(frame: torch.Tensor, w: int, h: int) -> torch.Tensor:
    array = frame.detach().cpu()
    channel_first = array.ndim == 3 and array.shape[0] in (1, 3)
    if channel_first:
        array = array.permute(1, 2, 0)

    np_frame = array.numpy()
    if np.issubdtype(np_frame.dtype, np.floating):
        max_value = float(np_frame.max()) if np_frame.size else 0.0
        if max_value <= 1.0:
            np_frame = np_frame * 255.0
        np_frame = np.clip(np_frame, 0, 255).round().astype(np.uint8)
    else:
        np_frame = np_frame.astype(np.uint8, copy=False)

    try:
        import cv2

        resized = cv2.resize(np_frame, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        mode = "L" if np_frame.ndim == 2 or np_frame.shape[-1] == 1 else None
        image = Image.fromarray(np_frame.squeeze(-1) if mode == "L" else np_frame, mode=mode)
        resized = np.asarray(image.resize((w, h), Image.Resampling.BOX), dtype=np.uint8)

    if resized.ndim == 2:
        resized = resized[..., None]
    out = torch.from_numpy(np.ascontiguousarray(resized))
    if channel_first:
        out = out.permute(2, 0, 1)
    return out
