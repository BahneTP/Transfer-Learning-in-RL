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

ATARI100K_GAMES = ["assault", "bankheist", "roadrunner", "breakout", "hero"]
ATARI100K_GAME_NAMES = {
    "assault": "Assault",
    "bankheist": "BankHeist",
    "roadrunner": "RoadRunner",
    "breakout": "Breakout",
    "hero": "Hero",
}
ATARI100K_ALGORITHMS = ["der", "spr", "sr_spr", "bbf", "sac_bbf"]
ATARI100K_TRANSFER_MODES = ["full", "linear", "attentive", "lora"]


@pytest.mark.parametrize(
    "experiment",
    [
        f"atari100k/{algorithm}/{game}"
        for algorithm in ATARI100K_ALGORITHMS
        for game in ATARI100K_GAMES
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


@pytest.mark.parametrize(
    "experiment",
    [
        f"atari100k/{algorithm}/{game}_resnet_{mode}"
        for algorithm in ("der", "bbf")
        for game in ATARI100K_GAMES
        for mode in ATARI100K_TRANSFER_MODES
    ],
)
def test_atari100k_resnet_transfer_experiment_configs_compose(experiment: str):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)
    game_key = experiment.split("/")[-1].split("_resnet_")[0]
    assert cfg.atari.game == ATARI100K_GAME_NAMES[game_key]
    assert cfg.algorithm.encoder_type == "resnet18"
    assert cfg.algorithm.resnet18_weights == "DEFAULT"
    assert cfg.algorithm.transfer_mode != "none"
    if "/bbf/" in experiment:
        assert cfg.algorithm.protect_encoder_from_reset is True


def test_smoke_atari100k_der_assault():
    """DER on Assault: tiny replay/update path through the real TorchRL adapter."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "atari100k/der/assault",
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
    cfg = load_experiment_cfg("atari100k/sr_spr/assault", BASE_OVERRIDES)

    assert cfg.algorithm.reset_every == 5_000
    assert cfg.algorithm.target_update_tau == 0.005
    assert cfg.algorithm.noisy is False
    assert cfg.algorithm.target_action_selection is True


def test_resnet18_encoder_forward_shape():
    from src.algorithms.atari100k.networks import RainbowDQNNetwork

    network = RainbowDQNNetwork(
        num_actions=4,
        num_atoms=51,
        noisy=False,
        dueling=True,
        distributional=True,
        encoder_type="resnet18",
        hidden_dim=128,
        input_channels=4,
    )
    support = torch.linspace(-10.0, 10.0, 51)
    output = network(
        torch.randint(0, 256, (2, 84, 84, 4), dtype=torch.uint8),
        support,
    )

    assert output.q_values.shape == (2, 4)
    assert output.logits is not None
    assert output.logits.shape == (2, 4, 51)
    assert output.latent.shape == (2, 512, 3, 3)


def test_der_train_step_with_resnet18_encoder():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="resnet18",
        hidden_dim=128,
        target_update_period=1,
        device="cpu",
    )
    agent = DERAgent(config, seed=7)
    batch = {
        "state": np.random.randint(0, 256, (2, 1, 84, 84, 4), dtype=np.uint8),
        "next_state": np.random.randint(0, 256, (2, 1, 84, 84, 4), dtype=np.uint8),
        "action": np.random.randint(0, 4, (2, 1), dtype=np.int32),
        "return": np.random.randn(2, 1).astype(np.float32),
        "terminal": np.zeros((2, 1), dtype=np.uint8),
        "discount": np.full((2, 1), 0.99, dtype=np.float32),
        "sampling_probabilities": np.ones((2,), dtype=np.float32),
    }

    metrics = agent.train_step(batch)

    assert metrics["TotalLoss"] >= 0.0
    assert metrics["priorities"].shape == (2,)


def test_linear_probe_freezes_encoder_and_uses_head_lr():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="resnet18",
        transfer_mode="linear_probe",
        encoder_lr_scale=0.1,
        hidden_dim=128,
        device="cpu",
    )
    agent = DERAgent(config, seed=3)

    assert all(not parameter.requires_grad for parameter in agent.online_network.encoder.parameters())
    assert any(
        parameter.requires_grad
        for name, parameter in agent.online_network.named_parameters()
        if name.startswith(("projection", "head"))
    )
    assert {group["lr"] for group in agent.optimizer.param_groups} == {1e-4}


def test_full_finetune_uses_scaled_encoder_lr():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="resnet18",
        transfer_mode="full_finetune",
        encoder_lr_scale=0.1,
        hidden_dim=128,
        device="cpu",
    )
    agent = DERAgent(config, seed=5)

    assert any(parameter.requires_grad for parameter in agent.online_network.encoder.parameters())
    assert {group["lr"] for group in agent.optimizer.param_groups} == {1e-5, 1e-4}


def test_attentive_probe_uses_attention_projection_and_freezes_encoder():
    from src.algorithms.atari100k.networks import AttentiveProbe
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="resnet18",
        transfer_mode="attentive_probe",
        hidden_dim=128,
        device="cpu",
    )
    agent = DERAgent(config, seed=11)

    assert isinstance(agent.online_network.projection, AttentiveProbe)
    assert all(not parameter.requires_grad for parameter in agent.online_network.encoder.parameters())
    output = agent.online_network(
        torch.randint(0, 256, (2, 84, 84, 4), dtype=torch.uint8),
        agent.support,
    )
    assert output.q_values.shape == (2, 4)


def test_lora_mode_trains_only_encoder_adapters_and_heads():
    from src.algorithms.atari100k.networks import LoRAConv2d
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="resnet18",
        transfer_mode="lora",
        hidden_dim=128,
        lora_rank=4,
        lora_alpha=8.0,
        target_update_period=1,
        device="cpu",
    )
    agent = DERAgent(config, seed=19)

    assert any(isinstance(module, LoRAConv2d) for module in agent.online_network.encoder.modules())
    trainable_encoder_names = [
        name for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_encoder_names
    assert all(".lora_" in name for name in trainable_encoder_names)
    assert any(
        parameter.requires_grad
        for name, parameter in agent.online_network.named_parameters()
        if name.startswith(("projection", "head"))
    )

    batch = {
        "state": np.random.randint(0, 256, (2, 1, 84, 84, 4), dtype=np.uint8),
        "next_state": np.random.randint(0, 256, (2, 1, 84, 84, 4), dtype=np.uint8),
        "action": np.random.randint(0, 4, (2, 1), dtype=np.int32),
        "return": np.random.randn(2, 1).astype(np.float32),
        "terminal": np.zeros((2, 1), dtype=np.uint8),
        "discount": np.full((2, 1), 0.99, dtype=np.float32),
    }

    metrics = agent.train_step(batch)

    assert metrics["TotalLoss"] >= 0.0
    assert metrics["priorities"].shape == (2,)


def test_lora_adapters_follow_agent_device_for_action_selection():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = DERConfig(
        num_actions=4,
        encoder_type="resnet18",
        transfer_mode="lora",
        hidden_dim=128,
        lora_rank=4,
        min_replay_history=0,
        device=device,
    )
    agent = DERAgent(config, seed=37)
    lora_parameters = [
        parameter
        for name, parameter in agent.online_network.encoder.named_parameters()
        if ".lora_" in name
    ]

    assert lora_parameters
    assert {parameter.device.type for parameter in lora_parameters} == {agent.device.type}
    action = agent.select_action(
        np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8),
        eval_mode=False,
    )
    assert action.device.type == agent.device.type


def test_static_transfer_metrics_include_parameter_counts():
    from src.algorithms.atari100k.algorithm import Atari100KAlgorithm
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    agent = DERAgent(
        DERConfig(
            num_actions=4,
            encoder_type="resnet18",
            transfer_mode="lora",
            hidden_dim=128,
            lora_rank=4,
            device="cpu",
        ),
        seed=31,
    )
    algorithm = Atari100KAlgorithm()
    algorithm.agent = agent

    metrics = algorithm._build_static_train_metrics()

    assert metrics["train/transfer_mode_lora"] == 1.0
    assert metrics["train/encoder_type_resnet18"] == 1.0
    assert metrics["train/params_total"] > 0
    assert metrics["train/params_trainable"] > 0
    assert metrics["train/params_encoder_total"] > metrics["train/params_encoder_trainable"]
    assert metrics["train/params_lora_trainable"] > 0


def test_freeze_encoder_bn_keeps_batch_norm_eval_after_train_step():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="resnet18",
        transfer_mode="full_finetune",
        freeze_encoder_bn=True,
        hidden_dim=128,
        target_update_period=1,
        device="cpu",
    )
    agent = DERAgent(config, seed=17)
    batch = {
        "state": np.random.randint(0, 256, (2, 1, 84, 84, 4), dtype=np.uint8),
        "next_state": np.random.randint(0, 256, (2, 1, 84, 84, 4), dtype=np.uint8),
        "action": np.random.randint(0, 4, (2, 1), dtype=np.int32),
        "return": np.random.randn(2, 1).astype(np.float32),
        "terminal": np.zeros((2, 1), dtype=np.uint8),
        "discount": np.full((2, 1), 0.99, dtype=np.float32),
    }

    agent.train_step(batch)
    batch_norms = [
        module for module in agent.online_network.encoder.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]

    assert batch_norms
    assert all(not module.training for module in batch_norms)
    assert all(
        not parameter.requires_grad
        for module in batch_norms
        for parameter in module.parameters()
    )


def test_bbf_protect_encoder_from_reset_skips_encoder_perturbation():
    from src.algorithms.atari100k.bbf import BBFAgent, BBFConfig

    config = BBFConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="resnet18",
        hidden_dim=128,
        reset_every=1,
        no_resets_after=100,
        protect_encoder_from_reset=True,
        target_update_period=1,
        device="cpu",
    )
    agent = BBFAgent(config, seed=23)
    before = {
        name: value.detach().clone()
        for name, value in agent.online_network.state_dict().items()
        if name.startswith("encoder.") and value.dtype.is_floating_point
    }

    agent.training_steps = 3
    agent.reset_weights()

    after = agent.online_network.state_dict()
    assert before
    assert all(torch.equal(value, after[name]) for name, value in before.items())


def test_bbf_lora_reset_keeps_adapter_state_dict_compatible():
    from src.algorithms.atari100k.bbf import BBFAgent, BBFConfig

    config = BBFConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="resnet18",
        transfer_mode="lora",
        hidden_dim=128,
        lora_rank=4,
        reset_every=1,
        no_resets_after=100,
        target_update_period=1,
        device="cpu",
    )
    agent = BBFAgent(config, seed=29)

    agent.training_steps = 3
    agent.reset_weights()

    trainable_encoder_names = [
        name for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_encoder_names
    assert all(".lora_" in name for name in trainable_encoder_names)


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
