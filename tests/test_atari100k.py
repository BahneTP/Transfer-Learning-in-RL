"""Smoke coverage for Atari 100K DER/SPR/BBF experiment wiring."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.conftest import load_experiment_cfg


BASE_OVERRIDES = [
    "logger=[]",
    "trainer.accelerator=cpu",
    "trainer.devices=[0]",
    "checkpoint.save_dir=/tmp/atari100k_smoke_tests/checkpoints",
    "checkpoint.save_last=false",
    "checkpoint.save_every_n_steps=999999999",
    "hydra.run.dir=/tmp/atari100k_smoke_tests",
]


@pytest.mark.parametrize(
    "experiment",
    [
        "atari100k/der/qbert",
        "atari100k/der/battlezone",
        "atari100k/spr/qbert",
        "atari100k/spr/battlezone",
        "atari100k/bbf/qbert",
        "atari100k/bbf/battlezone",
    ],
)
def test_atari100k_experiment_configs_compose(experiment: str):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)
    assert cfg.environment.name.startswith("ALE/")
    assert cfg.trainer.total_frames == 100_000
    assert cfg.algorithm.obs_key == "pixels"
    assert cfg.algorithm.seed == cfg.trainer.seed


def test_smoke_atari100k_der_qbert():
    """DER on Qbert: tiny replay/update path through the real TorchRL adapter."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "atari100k/der/qbert",
        [
            *BASE_OVERRIDES,
            "trainer.total_frames=20",
            "trainer.log_every_n_steps=10",
            "algorithm.replay_capacity=128",
            "algorithm.min_replay_history=8",
            "algorithm.batch_size=2",
            "algorithm.replay_ratio=2",
            "algorithm.frames_per_batch=1",
            "algorithm.update_horizon=1",
            "algorithm.epsilon_decay_period=8",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_atari100k_life_loss_is_stored_as_terminal():
    from src.algorithms.atari100k.algorithm import Atari100KAlgorithm

    class ReplaySpy:
        def __init__(self):
            self.sum_tree = SimpleNamespace(max_recorded_priority=1.0)
            self.terminal = None
            self.episode_end = None

        def add(
            self,
            observation,
            action,
            reward,
            terminal,
            *,
            priority,
            episode_end,
        ):
            self.terminal = terminal
            self.episode_end = episode_end

    algo = Atari100KAlgorithm()
    algo.replay = ReplaySpy()
    transition = TensorDict(
        {
            "pixels": torch.zeros(1, 84, 84, dtype=torch.uint8),
            "action": torch.tensor(2),
            "next": TensorDict(
                {
                    "reward": torch.tensor([0.0]),
                    "done": torch.tensor([False]),
                    "terminated": torch.tensor([False]),
                    "truncated": torch.tensor([False]),
                    "end-of-life": torch.tensor([True]),
                },
                batch_size=[],
            ),
        },
        batch_size=[],
    )

    algo._add_transition(transition)

    np.testing.assert_array_equal(algo.replay.terminal, np.array([1], dtype=np.uint8))
    np.testing.assert_array_equal(
        algo.replay.episode_end,
        np.array([1], dtype=np.uint8),
    )


def test_atari100k_life_loss_resets_policy_stack_flag():
    from src.algorithms.atari100k.algorithm import _is_init

    td = TensorDict(
        {
            "is_init": torch.tensor([False, False]),
            "end-of-life": torch.tensor([False, True]),
        },
        batch_size=[2],
    )

    np.testing.assert_array_equal(_is_init(td, 2), np.array([False, True]))


def test_episodic_life_reset_advances_without_resetting_game():
    import gymnasium as gym

    from src.environments.atari_wrappers import EpisodicLifeEnv

    class FakeAle:
        def __init__(self):
            self.current_lives = 3

        def lives(self):
            return self.current_lives

    class FakeEnv(gym.Env):
        action_space = gym.spaces.Discrete(3)
        observation_space = gym.spaces.Box(0, 255, (1,), dtype=np.uint8)
        metadata = {}
        render_mode = None

        def __init__(self):
            self.ale = FakeAle()
            self.reset_calls = 0
            self.actions = []

        @property
        def unwrapped(self):
            return self

        def reset(self, *, seed=None, options=None):
            self.reset_calls += 1
            return np.array([10], dtype=np.uint8), {}

        def step(self, action):
            self.actions.append(action)
            return np.array([20], dtype=np.uint8), 0.0, False, False, {}

    base = FakeEnv()
    env = EpisodicLifeEnv(base)
    env.reset()
    base.ale.current_lives = 2
    _, _, terminated, _, _ = env.step(1)
    assert terminated

    observation, _ = env.reset()

    assert base.reset_calls == 1
    assert base.actions == [1, 0]
    np.testing.assert_array_equal(observation, np.array([20], dtype=np.uint8))


def test_atari_algorithm_seed_controls_agent_and_replay():
    from src.algorithms.atari100k.algorithm import Atari100KAlgorithm

    algo = Atari100KAlgorithm(seed=123)
    assert algo.seed == 123


def test_atari_collector_keeps_environment_and_storage_on_cpu():
    from src.algorithms.atari100k.algorithm import Atari100KAlgorithm

    collector_cfg = Atari100KAlgorithm().get_collector_config()

    assert collector_cfg.env_device == "cpu"
    assert collector_cfg.policy_device == "cpu"
    assert collector_cfg.storing_device == "cpu"
