from __future__ import annotations

from typing import Any

from tqdm import tqdm


class ProgressCallback:
    """CLI progress bar for training, showing key metrics as postfix.

    Args:
        total_steps: total training frames/steps for the progress bar
    """

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self._bar: tqdm | None = None
        self._last_step: int = 0

    def on_train_start(self, state: dict[str, Any]) -> None:
        self._bar = tqdm(
            total=self.total_steps,
            unit="frames",
            dynamic_ncols=True,
            desc="Training",
        )
        self._last_step = 0

    _POSTFIX_SKIP_KEYS = ("time/speed", "time/step", "time/collect")
    _POSTFIX_PRIORITY_KEYS = ("train/raw_reward", "train/clip_reward")

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if self._bar is None:
            return
        delta = step - self._last_step
        self._bar.update(delta)
        self._last_step = step

        postfix: dict[str, str] = {}
        for key in self._POSTFIX_PRIORITY_KEYS:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                postfix[key] = f"{value:.4g}"

        for k, v in metrics.items():
            if (
                k in self._POSTFIX_PRIORITY_KEYS
                or k in self._POSTFIX_SKIP_KEYS
                or not isinstance(v, (int, float))
            ):
                continue
            postfix[k] = f"{v:.4g}"

        if postfix:
            self._bar.set_postfix(postfix)

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
