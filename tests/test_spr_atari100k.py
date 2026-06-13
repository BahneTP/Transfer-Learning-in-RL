from __future__ import annotations

import pytest
import torch
from hydra.utils import instantiate

from src.algorithms.atari100k.replay import PrioritizedNStepReplay
from tests.conftest import load_experiment_cfg


def test_prioritized_replay_samples_spr_sequences():
    replay = PrioritizedNStepReplay(
        capacity=32,
        observation_shape=(4, 84, 84),
        update_horizon=3,
        gamma=0.99,
        seed=0,
    )
    state = torch.zeros(4, 84, 84, dtype=torch.uint8)
    for i in range(12):
        replay.add(state + i, i % 4, float(i), i == 8, state + i + 1)
    batch = replay.sample_sequence(batch_size=4, device=torch.device("cpu"), jumps=5)
    assert batch.future_states is not None
    assert batch.rollout_actions is not None
    assert batch.same_trajectory is not None
    assert batch.future_states.shape == (4, 5, 4, 84, 84)
    assert batch.rollout_actions.shape == (4, 5)
    assert batch.same_trajectory.shape == (4, 5)


def test_spr_atari100k_config_instantiates():
    cfg = load_experiment_cfg(
        "spr/qbert_atari100k",
        [
            "logger=[]",
            "trainer.accelerator=cpu",
            "trainer.total_frames=10",
            "algorithm.replay_capacity=128",
            "algorithm.min_replay_history=8",
        ],
    )
    algorithm = instantiate(cfg.algorithm, device=None)
    assert algorithm.replay_capacity == 128
    assert algorithm.replay_ratio == 64
    assert algorithm.spr_weight == 5.0
    assert algorithm.jumps == 5


def test_smoke_spr_qbert_tiny():
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "spr/qbert_atari100k",
        [
            "logger=[]",
            "trainer.accelerator=cpu",
            "trainer.devices=[0]",
            "trainer.total_frames=20",
            "trainer.log_every_n_steps=10",
            "checkpoint.save_dir=/tmp/hydra_spr_atari100k_smoke/checkpoints",
            "checkpoint.save_last=false",
            "checkpoint.save_every_n_steps=999999999",
            "hydra.run.dir=/tmp/hydra_spr_atari100k_smoke",
            "algorithm.replay_capacity=128",
            "algorithm.min_replay_history=8",
            "algorithm.batch_size=4",
            "algorithm.replay_ratio=4",
            "algorithm.frames_per_batch=4",
            "algorithm.epsilon_decay_period=20",
            "algorithm.jumps=3",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert "train/spr_loss" in metrics
