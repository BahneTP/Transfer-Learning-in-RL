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
        "atari100k/sr_spr/qbert",
        "atari100k/sr_spr/battlezone",
        "atari100k/bbf/qbert",
        "atari100k/bbf/battlezone",
        "atari100k/sac_bbf/qbert",
        "atari100k/sac_bbf/battlezone",
    ],
)
def test_atari100k_experiment_configs_compose(experiment: str):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)
    assert cfg.environment.name.startswith("ALE/")
    assert cfg.trainer.total_frames == 100_000
    assert cfg.algorithm.obs_key == "pixels"
    assert cfg.trainer.eval_every_n_steps == 10_000
    assert cfg.trainer.num_eval_episodes == 10
    assert cfg.trainer.final_num_eval_episodes == 20
    assert cfg.algorithm.seed == cfg.trainer.seed


def test_smoke_atari100k_der_qbert():
    """DER on Qbert: tiny replay/update path through the real TorchRL adapter."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "atari100k/der/qbert",
        [
            *BASE_OVERRIDES,
            "trainer.eval_every_n_steps=null",
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


@pytest.mark.parametrize(
    ("algorithm_class", "config_class", "expected"),
    [
        ("DERAlgorithm", "DERConfig", False),
        ("SPRAlgorithm", "SPRConfig", True),
        ("SRSPRAlgorithm", "SRSPRConfig", False),
        ("BBFAlgorithm", "BBFConfig", False),
        ("SACBBFAlgorithm", "SACBBFConfig", False),
    ],
)
def test_replay_prefetch_is_enabled_only_for_fixed_schedule_spr(
    algorithm_class: str,
    config_class: str,
    expected: bool,
):
    from src.algorithms.atari100k import algorithm as algorithm_module
    from src.algorithms.atari100k import bbf, der, sac_bbf, spr

    config_modules = {
        "DERConfig": der,
        "SPRConfig": spr,
        "SRSPRConfig": spr,
        "BBFConfig": bbf,
        "SACBBFConfig": sac_bbf,
    }
    algorithm = getattr(algorithm_module, algorithm_class)(
        device=torch.device("cuda")
    )
    config_kwargs = {"num_actions": 4}
    if config_class == "SPRConfig":
        config_kwargs["cycle_steps"] = 0
    config = getattr(config_modules[config_class], config_class)(**config_kwargs)
    algorithm.agent = SimpleNamespace(config=config)

    assert algorithm._can_prefetch_replay() is expected


def test_spr_prefetch_reuses_the_background_sample():
    from src.algorithms.atari100k.algorithm import SPRAlgorithm

    class ReplaySpy:
        def __init__(self):
            self.samples = 0

        def sample_transition_batch(self, **kwargs):
            self.samples += 1
            return {
                "indices": np.array([self.samples], dtype=np.int32),
            }

        def set_priority(self, indices, priorities):
            return None

    class AgentSpy:
        config = SimpleNamespace(
            spr_weight=5.0,
            cycle_steps=0,
            replay_ratio=64,
            batch_size=32,
            batches_to_group=2,
        )
        reset_priorities_requested = False

        def train_step(self, batch):
            return {
                "TotalLoss": 0.0,
                "priorities": np.ones_like(batch["indices"], dtype=np.float32),
            }

    algorithm = SPRAlgorithm(device=torch.device("cuda"))
    algorithm.agent = AgentSpy()
    algorithm.replay = ReplaySpy()
    try:
        algorithm._train_step_updates()
        algorithm._finish_replay_prefetch_before_add()
        assert algorithm.replay.samples == 2

        algorithm._train_step_updates()
        algorithm._finish_replay_prefetch_before_add()
        assert algorithm.replay.samples == 3
    finally:
        if algorithm._sample_executor is not None:
            algorithm._sample_executor.shutdown()


def test_sr_spr_reference_preset_values_compose():
    cfg = load_experiment_cfg("atari100k/sr_spr/qbert", BASE_OVERRIDES)

    assert cfg.algorithm.reset_every == 5_000
    assert cfg.algorithm.target_update_tau == 0.005
    assert cfg.algorithm.noisy is False
    assert cfg.algorithm.target_action_selection is True


def test_sac_bbf_train_step_includes_policy_metrics():
    from src.algorithms.atari100k.sac_bbf import SACBBFAgent, SACBBFConfig

    config = SACBBFConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="dqn",
        hidden_dim=128,
        width_scale=1,
        jumps=3,
        spr_weight=1.0,
        reset_every=20,
        no_resets_after=100,
        target_update_period=1,
        device="cpu",
    )
    agent = SACBBFAgent(config, seed=13)
    batch = {
        "state": np.random.randint(0, 256, (2, 4, 84, 84, 4), dtype=np.uint8),
        "next_state": np.random.randint(0, 256, (2, 4, 84, 84, 4), dtype=np.uint8),
        "action": np.random.randint(0, 4, (2, 4), dtype=np.int32),
        "return": np.random.randn(2, 4).astype(np.float32),
        "terminal": np.zeros((2, 4), dtype=np.uint8),
        "discount": np.full((2, 4), 0.99, dtype=np.float32),
        "same_trajectory": np.ones((2, 4), dtype=np.uint8),
        "sampling_probabilities": np.ones((2,), dtype=np.float32),
    }

    metrics = agent.train_step(batch)

    assert "PolicyLoss" in metrics
    assert "Entropy" in metrics
    assert metrics["PolicySampleActionHistogram"].shape == (4,)
