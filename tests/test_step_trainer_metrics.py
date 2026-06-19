from __future__ import annotations

import torch
from tensordict import TensorDict

from src.trainers.StepTrainer import _batch_metrics, _evaluation_episode_count


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


def test_evaluation_episode_count_uses_regular_and_final_sizes():
    common = {
        "batch_frames": 1,
        "total_frames": 100_000,
        "eval_every": 10_000,
        "eval_episodes": 10,
        "final_eval_episodes": 20,
    }

    assert _evaluation_episode_count(step=9_999, **common) is None
    assert _evaluation_episode_count(step=10_000, **common) == 10
    assert _evaluation_episode_count(step=90_000, **common) == 10
    assert _evaluation_episode_count(step=100_000, **common) == 20


def test_evaluation_episode_count_can_be_disabled():
    assert _evaluation_episode_count(
        step=100_000,
        batch_frames=1,
        total_frames=100_000,
        eval_every=0,
        eval_episodes=10,
        final_eval_episodes=20,
    ) is None
