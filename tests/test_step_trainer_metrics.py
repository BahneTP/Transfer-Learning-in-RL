from __future__ import annotations

import torch
from tensordict import TensorDict

from src.trainers.StepTrainer import _batch_metrics


def test_batch_metrics_prefers_raw_episode_reward_when_available():
    batch = TensorDict(
        {
            "next": {
                "done": torch.tensor([False, True, False, True]),
                "episode_reward": torch.tensor([0.0, 1.0, 0.0, -1.0]),
                "raw_episode_reward": torch.tensor([0.0, 100.0, 0.0, 25.0]),
                "step_count": torch.tensor([0.0, 11.0, 0.0, 13.0]),
            }
        },
        batch_size=[4],
    )

    metrics = _batch_metrics(batch)

    assert metrics["train/raw_reward"] == 62.5
    assert metrics["train/clip_reward"] == 0.0
    assert metrics["train/episode_length"] == 12.0


def test_batch_metrics_falls_back_to_clipped_episode_reward():
    batch = TensorDict(
        {
            "next": {
                "done": torch.tensor([False, True]),
                "episode_reward": torch.tensor([0.0, 3.0]),
                "step_count": torch.tensor([0.0, 9.0]),
            }
        },
        batch_size=[2],
    )

    metrics = _batch_metrics(batch)

    assert metrics["train/raw_reward"] == 3.0
    assert "train/clip_reward" not in metrics
    assert metrics["train/episode_length"] == 9.0
