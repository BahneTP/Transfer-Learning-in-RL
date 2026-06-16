from __future__ import annotations

import pytest
import numpy as np
import torch
from hydra.utils import instantiate
from tensordict import TensorDict

from src.algorithms.atari100k import RainbowAtari100KAlgorithm
from src.algorithms.atari100k.networks import NoisyLinear
from src.algorithms.atari100k.replay import PrioritizedNStepReplay
from tests.conftest import load_experiment_cfg


def test_prioritized_n_step_replay_samples_expected_shapes():
    replay = PrioritizedNStepReplay(
        capacity=32,
        observation_shape=(4, 84, 84),
        update_horizon=3,
        gamma=0.99,
        seed=0,
    )
    state = torch.zeros(4, 84, 84, dtype=torch.uint8)
    for i in range(8):
        replay.add(state + i, i % 4, float(i), i == 6, state + i + 1)
    batch = replay.sample(batch_size=4, device=torch.device("cpu"))
    assert batch.states.shape == (4, 4, 84, 84)
    assert batch.next_states.shape == (4, 4, 84, 84)
    assert batch.actions.shape == (4,)
    assert batch.returns.shape == (4,)
    assert batch.weights.shape == (4,)


def test_prioritized_replay_stores_priority_values_directly():
    replay = PrioritizedNStepReplay(
        capacity=32,
        observation_shape=(4, 84, 84),
        update_horizon=3,
        gamma=0.99,
        seed=0,
    )
    replay.set_priority(np.array([3]), np.array([0.25], dtype=np.float32))
    assert replay.sum_tree.get(np.array([3])).item() == pytest.approx(0.25)


def test_replay_uint8_frames_are_not_unit_scaled():
    replay = PrioritizedNStepReplay(
        capacity=32,
        observation_shape=(4, 2, 2),
        update_horizon=3,
        gamma=0.99,
        seed=0,
    )
    state = torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2)
    frame = replay._latest_frame_uint8(state)
    assert frame.tolist() == state[-1].numpy().tolist()


def test_rainbow_atari100k_config_instantiates():
    cfg = load_experiment_cfg(
        "rainbow/qbert_atari100k",
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
    assert algorithm.min_replay_history == 8
    assert algorithm.replay_ratio == 32
    assert algorithm.eval_noise is True
    assert algorithm.target_eval_mode is False
    assert algorithm.target_update_tau == 1.0


def test_rainbow_stores_end_of_life_as_replay_terminal():
    algorithm = RainbowAtari100KAlgorithm(
        device=torch.device("cpu"),
        replay_capacity=32,
        min_replay_history=8,
    )
    algorithm.replay = PrioritizedNStepReplay(
        capacity=32,
        observation_shape=(4, 84, 84),
        update_horizon=3,
        gamma=0.99,
        seed=0,
    )
    state = torch.zeros(1, 4, 84, 84)
    batch = TensorDict(
        {
            "pixels": state,
            "action": torch.tensor([1]),
            "next": {
                "pixels": state,
                "reward": torch.tensor([[0.0]]),
                "done": torch.tensor([[False]]),
                "end-of-life": torch.tensor([[True]]),
            },
        },
        batch_size=[1],
    )
    algorithm._store_batch(batch)
    assert algorithm.replay.dones[0]


def test_rainbow_stores_scalar_action_with_extra_dim():
    algorithm = RainbowAtari100KAlgorithm(
        device=torch.device("cpu"),
        replay_capacity=32,
        min_replay_history=8,
    )
    algorithm.num_actions = 6
    algorithm.replay = PrioritizedNStepReplay(
        capacity=32,
        observation_shape=(4, 84, 84),
        update_horizon=3,
        gamma=0.99,
        seed=0,
    )
    state = torch.zeros(1, 4, 84, 84)
    batch = TensorDict(
        {
            "pixels": state,
            "action": torch.tensor([[3]]),
            "next": {
                "pixels": state,
                "reward": torch.tensor([[0.0]]),
                "done": torch.tensor([[False]]),
            },
        },
        batch_size=[1],
    )
    algorithm._store_batch(batch)
    assert algorithm.replay.actions[0] == 3


def test_noisy_linear_resamples_noise_per_forward():
    layer = NoisyLinear(8, 4)
    x = torch.ones(2, 8)
    train_a = layer(x, eval_mode=False)
    train_b = layer(x, eval_mode=False)
    eval_a = layer(x, eval_mode=True)
    eval_b = layer(x, eval_mode=True)
    assert not torch.equal(train_a, train_b)
    assert torch.equal(eval_a, eval_b)


def test_smoke_rainbow_qbert_tiny():
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "rainbow/qbert_atari100k",
        [
            "logger=[]",
            "trainer.accelerator=cpu",
            "trainer.devices=[0]",
            "trainer.total_frames=20",
            "trainer.log_every_n_steps=10",
            "checkpoint.save_dir=/tmp/hydra_rainbow_atari100k_smoke/checkpoints",
            "checkpoint.save_last=false",
            "checkpoint.save_every_n_steps=999999999",
            "hydra.run.dir=/tmp/hydra_rainbow_atari100k_smoke",
            "algorithm.replay_capacity=128",
            "algorithm.min_replay_history=8",
            "algorithm.batch_size=4",
            "algorithm.replay_ratio=4",
            "algorithm.frames_per_batch=4",
            "algorithm.epsilon_decay_period=20",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
