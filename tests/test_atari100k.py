"""Smoke coverage for Atari 100K DER/SPR/BBF experiment wiring."""
from __future__ import annotations

from pathlib import Path
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

ATARI100K_GAMES = ["assault", "bankheist", "roadrunner", "breakout", "hero", "jamesbond"]
ATARI100K_GAME_NAMES = {
    "assault": "Assault",
    "bankheist": "BankHeist",
    "roadrunner": "RoadRunner",
    "breakout": "Breakout",
    "hero": "Hero",
    "jamesbond": "Jamesbond",
}
ATARI100K_ALGORITHMS = ["der", "spr", "sr_spr", "bbf", "sac_bbf"]
ATARI100K_TRANSFER_MODES = ["full", "linear", "attentive", "lora"]
TRANSFER_GAMES = ["assault", "bankheist", "jamesbond", "roadrunner"]
TRANSFER_ALGORITHMS = ["der", "sac_bbf"]
DINO_TRANSFER_MODES = {
    "full": "full_finetune",
    "linear": "linear_probe",
    "attentive": "attentive_probe",
    "lora": "lora",
}
DINO_BLOCK_EXPERIMENTS = {
    "full_block3": ("full_finetune", 3),
    "full_block6": ("full_finetune", 6),
    "full_block9": ("full_finetune", 9),
    "linear_block1": ("linear_probe", 1),
    "linear_block2": ("linear_probe", 2),
    "linear_block3": ("linear_probe", 3),
    "linear_block4": ("linear_probe", 4),
    "linear_block5": ("linear_probe", 5),
    "linear_block6": ("linear_probe", 6),
    "linear_block7": ("linear_probe", 7),
    "linear_block8": ("linear_probe", 8),
    "linear_block9": ("linear_probe", 9),
    "linear_block10": ("linear_probe", 10),
    "linear_block11": ("linear_probe", 11),
}
DINO_VITS14_WEIGHTS = Path("models/dinov2_vits14_pretrain.pth")


def optimizer_lrs_by_parameter_name(agent) -> dict[str, float]:
    parameter_names = {
        parameter: name
        for name, parameter in agent.online_network.named_parameters()
    }
    lrs = {}
    for group in agent.optimizer.param_groups:
        for parameter in group["params"]:
            lrs[parameter_names[parameter]] = group["lr"]
    return lrs


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
    assert cfg.trainer.num_eval_episodes == 50
    assert cfg.algorithm.seed == cfg.trainer.seed


@pytest.mark.parametrize(
    "experiment",
    [
        f"dinov2/{algorithm}/{mode}/{game}"
        for algorithm in TRANSFER_ALGORITHMS
        for mode in DINO_TRANSFER_MODES
        for game in TRANSFER_GAMES
    ],
)
def test_dinov2_transfer_experiment_configs_compose(experiment: str):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)
    _, algorithm, mode, game = experiment.split("/")

    assert cfg.atari.game == ATARI100K_GAME_NAMES[game]
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.dinov2_weights == "models/dinov2_vits14_pretrain.pth"
    assert cfg.algorithm.transfer_mode == DINO_TRANSFER_MODES[mode]
    assert cfg.algorithm.learning_rate == pytest.approx(1e-4)
    if mode == "full":
        assert cfg.algorithm.encoder_lr_scale == pytest.approx(0.01)
    if algorithm == "sac_bbf":
        assert cfg.algorithm.protect_encoder_from_reset is True


@pytest.mark.parametrize(
    "experiment",
    [f"dinov2/der/{mode}/jamesbond" for mode in DINO_BLOCK_EXPERIMENTS],
)
def test_dinov2_block_transfer_experiment_configs_compose(experiment: str):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)
    _, _, mode, game = experiment.split("/")
    transfer_mode, output_block = DINO_BLOCK_EXPERIMENTS[mode]

    assert game == "jamesbond"
    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == transfer_mode
    assert cfg.algorithm.dinov2_output_block == output_block


def test_dinov2_layer_mix_transfer_experiment_config_compose():
    cfg = load_experiment_cfg("dinov2/der/linear_mix/jamesbond", BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == "linear_probe"
    assert cfg.algorithm.dinov2_output_mode == "layer_mix"
    assert list(cfg.algorithm.dinov2_mix_blocks) == list(range(1, 13))


@pytest.mark.parametrize(
    ("experiment", "mix_blocks"),
    [
        ("dinov2/der/linear_mix_blocks_1_5/jamesbond", [1, 2, 3, 4, 5]),
        ("dinov2/der/linear_mix_blocks_8_12/jamesbond", [8, 9, 10, 11, 12]),
        ("dinov2/der/linear_mix_blocks_3_7_11/jamesbond", [3, 7, 11]),
    ],
)
def test_dinov2_layer_mix_subset_experiment_configs_compose(
    experiment: str,
    mix_blocks: list[int],
):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == "linear_probe"
    assert cfg.algorithm.dinov2_output_mode == "layer_mix"
    assert list(cfg.algorithm.dinov2_mix_blocks) == mix_blocks


@pytest.mark.parametrize(
    ("experiment", "rank", "alpha"),
    [
        ("dinov2/der/lora_block3_r1_a2/jamesbond", 1, 2.0),
        ("dinov2/der/lora_block3_r2_a4/jamesbond", 2, 4.0),
        ("dinov2/der/lora_block3_r4_a8/jamesbond", 4, 8.0),
        ("dinov2/der/lora_block3_r8_a16/jamesbond", 8, 16.0),
        ("dinov2/der/lora_block3_r16_a32/jamesbond", 16, 32.0),
    ],
)
def test_dinov2_lora_block3_rank_alpha_experiment_configs_compose(
    experiment: str,
    rank: int,
    alpha: float,
):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == "lora"
    assert cfg.algorithm.dinov2_output_block == 3
    assert cfg.algorithm.lora_rank == rank
    assert cfg.algorithm.lora_alpha == pytest.approx(alpha)


def test_dinov2_jepa_block3_experiment_config_compose():
    cfg = load_experiment_cfg("dinov2/der/jepa_block3/jamesbond", BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == "jepa_probe"
    assert cfg.algorithm.dinov2_output_block == 3
    assert cfg.algorithm.jepa_loss_weight == pytest.approx(1.0)
    assert cfg.algorithm.jepa_action_dim == 64
    assert cfg.algorithm.jepa_prediction_mode == "direct"


def test_dinov2_jepa_full_block3_experiment_config_compose():
    cfg = load_experiment_cfg("dinov2/der/jepa_full_block3/jamesbond", BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == "jepa_full_finetune"
    assert cfg.algorithm.dinov2_output_block == 3
    assert cfg.algorithm.learning_rate == pytest.approx(1e-4)
    assert cfg.algorithm.encoder_lr_scale == pytest.approx(1.0)
    assert cfg.algorithm.jepa_loss_weight == pytest.approx(1.0)
    assert cfg.algorithm.jepa_action_dim == 64
    assert cfg.algorithm.jepa_prediction_mode == "direct"
    assert cfg.algorithm.temporal_straightening_weight == pytest.approx(0.0)
    assert cfg.algorithm.lambda_sigreg == pytest.approx(0.0)


def test_dinov2_jepa_full_straight_block3_experiment_config_compose():
    cfg = load_experiment_cfg("dinov2/der/jepa_full_straight_block3/jamesbond", BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == "jepa_full_finetune"
    assert cfg.algorithm.dinov2_output_block == 3
    assert cfg.algorithm.encoder_lr_scale == pytest.approx(1.0)
    assert cfg.algorithm.jepa_prediction_mode == "residual"
    assert cfg.algorithm.temporal_straightening_weight == pytest.approx(0.1)


def test_dinov2_jepa_full_sigreg_block3_experiment_config_compose():
    cfg = load_experiment_cfg("dinov2/der/jepa_full_sigreg_block3/jamesbond", BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.transfer_mode == "jepa_full_finetune"
    assert cfg.algorithm.dinov2_output_block == 3
    assert cfg.algorithm.jepa_loss_weight == pytest.approx(1.0)
    assert cfg.algorithm.jepa_prediction_mode == "residual"
    assert cfg.algorithm.lambda_sigreg == pytest.approx(0.1)
    assert cfg.algorithm.temporal_straightening_weight == pytest.approx(0.0)


@pytest.mark.parametrize(
    "experiment",
    [
        f"resnet/{algorithm}/{mode}/{game}"
        for algorithm in TRANSFER_ALGORITHMS
        for mode in DINO_TRANSFER_MODES
        for game in TRANSFER_GAMES
    ],
)
def test_resnet_transfer_experiment_configs_compose(experiment: str):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)
    _, algorithm, mode, game = experiment.split("/")

    assert cfg.atari.game == ATARI100K_GAME_NAMES[game]
    assert cfg.algorithm.encoder_type == "resnet18"
    assert cfg.algorithm.resnet18_weights == "DEFAULT"
    assert cfg.algorithm.resnet18_variant == "resnet_layer3_reduced"
    assert cfg.algorithm.transfer_mode == DINO_TRANSFER_MODES[mode]
    if algorithm == "sac_bbf":
        assert cfg.algorithm.protect_encoder_from_reset is True


@pytest.mark.parametrize(
    ("experiment", "variant"),
    [
        ("resnet/der/linear_layer1/jamesbond", "resnet_layer1_reduced"),
        ("resnet/der/linear_layer2/jamesbond", "resnet_layer2_reduced"),
        ("resnet/der/linear_layer3/jamesbond", "resnet_layer3_reduced"),
        ("resnet/der/linear_layer4/jamesbond", "resnet_layer4_reduced"),
    ],
)
def test_resnet_linear_layer_experiment_configs_compose(experiment: str, variant: str):
    cfg = load_experiment_cfg(experiment, BASE_OVERRIDES)

    assert cfg.atari.game == "Jamesbond"
    assert cfg.algorithm.encoder_type == "resnet18"
    assert cfg.algorithm.resnet18_weights == "DEFAULT"
    assert cfg.algorithm.resnet18_variant == variant
    assert cfg.algorithm.transfer_mode == "linear_probe"


def test_atari100k_dinov2_transfer_override_compose():
    cfg = load_experiment_cfg(
        "atari100k/der/jamesbond",
        [
            *BASE_OVERRIDES,
            "algorithm.encoder_type=dinov2_vits14",
            "algorithm.dinov2_weights=models/dinov2_vits14_pretrain.pth",
            "algorithm.transfer_mode=linear_probe",
        ],
    )

    assert cfg.algorithm.encoder_type == "dinov2_vits14"
    assert cfg.algorithm.dinov2_weights == "models/dinov2_vits14_pretrain.pth"
    assert cfg.algorithm.transfer_mode == "linear_probe"


def test_smoke_atari100k_der_assault():
    """DER on Assault: tiny replay/update path through the real TorchRL adapter."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "atari100k/der/assault",
        [
            *BASE_OVERRIDES,
            "trainer.total_frames=20",
            "trainer.log_every_n_steps=10",
            "trainer.num_eval_episodes=0",
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


def test_step_trainer_accumulates_episode_metrics_until_flush():
    from collections import defaultdict

    from src.trainers.StepTrainer import _accumulate_episode_metrics
    from src.trainers.StepTrainer import _flush_episode_metrics

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    first = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": torch.tensor([True, False]),
                    "raw_episode_reward": torch.tensor([10.0, 0.0]),
                    "episode_reward": torch.tensor([1.0, 0.0]),
                    "step_count": torch.tensor([100, 0]),
                },
                batch_size=[2],
            )
        },
        batch_size=[2],
    )
    second = TensorDict(
        {
            "next": TensorDict(
                {
                    "done": torch.tensor([True, True]),
                    "raw_episode_reward": torch.tensor([20.0, 30.0]),
                    "episode_reward": torch.tensor([2.0, 3.0]),
                    "step_count": torch.tensor([200, 300]),
                },
                batch_size=[2],
            )
        },
        batch_size=[2],
    )

    _accumulate_episode_metrics(first, sums, counts)
    _accumulate_episode_metrics(second, sums, counts)
    metrics = _flush_episode_metrics(sums, counts)

    assert metrics["train/raw_reward"] == pytest.approx(20.0)
    assert metrics["train/clip_reward"] == pytest.approx(2.0)
    assert metrics["train/episode_length"] == pytest.approx(200.0)
    assert metrics["train/episodes"] == pytest.approx(3.0)
    assert not sums
    assert not counts


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
    assert output.latent.shape == (2, 64, 6, 6)
    assert network.encoder.input_adapter.in_channels == 4
    assert network.encoder.input_adapter.out_channels == 3
    assert network.encoder.stem[0].in_channels == 3


@pytest.mark.parametrize(
    ("variant", "expected_shape"),
    [
        ("resnet_full", (2, 512, 3, 3)),
        ("resnet_layer1_reduced", (2, 16, 21, 21)),
        ("resnet_layer2_reduced", (2, 32, 11, 11)),
        ("resnet_layer3_flattened", (2, 256, 6, 6)),
        ("resnet_layer3_reduced", (2, 64, 6, 6)),
        ("resnet_layer4_reduced", (2, 128, 3, 3)),
    ],
)
def test_resnet18_variants_define_spatial_feature_shape(variant: str, expected_shape: tuple[int, ...]):
    from src.algorithms.atari100k.networks import RainbowDQNNetwork

    network = RainbowDQNNetwork(
        num_actions=4,
        num_atoms=51,
        noisy=False,
        dueling=True,
        distributional=True,
        encoder_type="resnet18",
        resnet18_variant=variant,  # type: ignore[arg-type]
        hidden_dim=128,
        input_channels=4,
    )

    latent = network.encode(torch.randint(0, 256, (2, 84, 84, 4), dtype=torch.uint8))

    assert latent.shape == expected_shape
    assert network.encoder.input_adapter.in_channels == 4
    assert network.encoder.input_adapter.out_channels == 3
    assert network.encoder.stem[0].in_channels == 3


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_vits14_encoder_forward_shape():
    from src.algorithms.atari100k.networks import RainbowDQNNetwork

    network = RainbowDQNNetwork(
        num_actions=4,
        num_atoms=51,
        noisy=False,
        dueling=True,
        distributional=True,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        hidden_dim=128,
        input_channels=4,
    )

    latent = network.encode(torch.randint(0, 256, (1, 84, 84, 4), dtype=torch.uint8))

    assert latent.shape == (1, 64, 6, 6)


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
@pytest.mark.parametrize("output_block", [6, 9])
def test_dinov2_vits14_output_block_forward_shape(output_block: int):
    from src.algorithms.atari100k.networks import RainbowDQNNetwork

    network = RainbowDQNNetwork(
        num_actions=4,
        num_atoms=51,
        noisy=False,
        dueling=True,
        distributional=True,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        dinov2_output_block=output_block,
        hidden_dim=128,
        input_channels=4,
    )

    latent = network.encode(torch.randint(0, 256, (1, 84, 84, 4), dtype=torch.uint8))

    assert network.encoder.output_block == output_block
    assert latent.shape == (1, 64, 6, 6)


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_vits14_layer_mix_forward_shape():
    from src.algorithms.atari100k.networks import RainbowDQNNetwork

    network = RainbowDQNNetwork(
        num_actions=4,
        num_atoms=51,
        noisy=False,
        dueling=True,
        distributional=True,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        dinov2_output_mode="layer_mix",
        dinov2_mix_blocks=tuple(range(1, 13)),
        hidden_dim=128,
        input_channels=4,
    )

    latent = network.encode(torch.randint(0, 256, (1, 84, 84, 4), dtype=torch.uint8))
    weights = network.encoder.layer_mix_weights()

    assert latent.shape == (1, 64, 6, 6)
    assert weights.shape == (12,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    assert torch.allclose(weights, torch.full((12,), 1.0 / 12.0))


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

    trainable_encoder_params = [
        name
        for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_encoder_params == [
        "input_adapter.weight",
        "input_adapter.bias",
        "reducer.weight",
        "reducer.bias",
    ]
    assert any(
        parameter.requires_grad
        for name, parameter in agent.online_network.named_parameters()
        if name.startswith(("projection", "head"))
    )
    assert {group["lr"] for group in agent.optimizer.param_groups} == {1e-4}
    lrs = optimizer_lrs_by_parameter_name(agent)
    assert lrs["encoder.input_adapter.weight"] == 1e-4
    assert lrs["encoder.input_adapter.bias"] == 1e-4
    assert lrs["encoder.reducer.weight"] == 1e-4
    assert lrs["encoder.reducer.bias"] == 1e-4


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_linear_probe_keeps_adapter_and_reducer_trainable():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        transfer_mode="linear_probe",
        hidden_dim=128,
        device="cpu",
    )
    agent = DERAgent(config, seed=13)

    trainable_encoder_params = [
        name
        for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable_encoder_params == [
        "input_adapter.weight",
        "input_adapter.bias",
        "reducer.weight",
        "reducer.bias",
    ]


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_linear_mix_probe_keeps_mix_logits_trainable():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        dinov2_output_mode="layer_mix",
        dinov2_mix_blocks=tuple(range(1, 13)),
        transfer_mode="linear_probe",
        hidden_dim=128,
        device="cpu",
    )
    agent = DERAgent(config, seed=17)

    trainable_encoder_params = [
        name
        for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]
    lrs = optimizer_lrs_by_parameter_name(agent)

    assert trainable_encoder_params == [
        "mix_logits",
        "input_adapter.weight",
        "input_adapter.bias",
        "reducer.weight",
        "reducer.bias",
    ]
    assert lrs["encoder.mix_logits"] == 1e-4
    assert lrs["encoder.input_adapter.weight"] == 1e-4
    assert lrs["encoder.reducer.weight"] == 1e-4


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_layer_mix_metrics_are_reported():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        dinov2_output_mode="layer_mix",
        dinov2_mix_blocks=tuple(range(1, 13)),
        transfer_mode="linear_probe",
        hidden_dim=128,
        device="cpu",
    )
    agent = DERAgent(config, seed=19)

    metrics = agent._transfer_metrics()

    assert sorted(metrics) == [
        f"dinov2_layer_mix/block_{block:02d}" for block in range(1, 13)
    ]
    assert sum(metrics.values()) == pytest.approx(1.0)
    assert metrics["dinov2_layer_mix/block_01"] == pytest.approx(1.0 / 12.0)


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_jepa_probe_keeps_backbone_frozen_and_trains_predictor():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        dinov2_output_block=3,
        transfer_mode="jepa_probe",
        hidden_dim=128,
        jepa_action_dim=16,
        device="cpu",
    )
    agent = DERAgent(config, seed=23)

    trainable_encoder_params = [
        name
        for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]
    trainable_jepa_params = [
        name
        for name, parameter in agent.online_network.named_parameters()
        if name.startswith(("jepa_action_embedding", "jepa_predictor")) and parameter.requires_grad
    ]
    lrs = optimizer_lrs_by_parameter_name(agent)

    assert trainable_encoder_params == [
        "input_adapter.weight",
        "input_adapter.bias",
        "reducer.weight",
        "reducer.bias",
    ]
    assert trainable_jepa_params == [
        "jepa_action_embedding.weight",
        "jepa_predictor.0.weight",
        "jepa_predictor.0.bias",
        "jepa_predictor.2.weight",
        "jepa_predictor.2.bias",
    ]
    assert lrs["jepa_action_embedding.weight"] == 1e-4
    assert lrs["jepa_predictor.0.weight"] == 1e-4


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_jepa_probe_train_step_reports_jepa_loss():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        dinov2_output_block=3,
        transfer_mode="jepa_probe",
        hidden_dim=128,
        jepa_action_dim=16,
        target_update_period=1,
        device="cpu",
    )
    agent = DERAgent(config, seed=29)
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

    assert metrics["TotalLoss"] >= metrics["DQNLoss"]
    assert metrics["JEPALoss"] > 0.0
    assert metrics["priorities"].shape == (2,)


@pytest.mark.skipif(not DINO_VITS14_WEIGHTS.exists(), reason="DINOv2 weights are not available locally")
def test_dinov2_jepa_full_finetune_trains_backbone_and_predictor():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dinov2_vits14",
        dinov2_weights=str(DINO_VITS14_WEIGHTS),
        dinov2_output_block=3,
        transfer_mode="jepa_full_finetune",
        hidden_dim=128,
        jepa_action_dim=16,
        device="cpu",
    )
    agent = DERAgent(config, seed=30)
    lrs = optimizer_lrs_by_parameter_name(agent)

    assert any(
        name.startswith("blocks.0.")
        for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    )
    assert lrs["encoder.input_adapter.weight"] == pytest.approx(1e-4)
    assert lrs["encoder.reducer.weight"] == pytest.approx(1e-4)
    assert lrs["jepa_predictor.0.weight"] == pytest.approx(1e-4)
    assert lrs["encoder.blocks.0.norm1.weight"] == pytest.approx(1e-4)


def test_jepa_residual_prediction_adds_current_latent():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dqn",
        transfer_mode="jepa_probe",
        jepa_prediction_mode="residual",
        hidden_dim=4,
        jepa_action_dim=4,
        device="cpu",
    )
    agent = DERAgent(config, seed=31)
    current_latent = torch.tensor([[1.0, 2.0, 4.0, 8.0], [0.5, 1.5, 2.5, 3.5]])
    predicted_update = torch.tensor([[0.0, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, -0.5]])
    target_next = current_latent + predicted_update
    calls = {"encode": 0}

    def fake_encode_jepa_latent(states, *, eval_mode=False):
      calls["encode"] += 1
      return current_latent if calls["encode"] == 1 else target_next

    def fake_predict_next_jepa_latent(latent, actions, *, action_dim):
      assert latent is current_latent
      assert action_dim == 4
      return predicted_update

    agent.online_network.encode_jepa_latent = fake_encode_jepa_latent
    agent.online_network.predict_next_jepa_latent = fake_predict_next_jepa_latent

    loss = agent._jepa_loss(
        torch.zeros((2, 4, 84, 84), dtype=torch.uint8),
        torch.zeros((2, 4, 84, 84), dtype=torch.uint8),
        torch.tensor([0, 1]),
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_temporal_straightening_loss_uses_three_consecutive_latents():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dqn",
        transfer_mode="jepa_full_finetune",
        temporal_straightening_weight=0.1,
        hidden_dim=4,
        device="cpu",
    )
    agent = DERAgent(config, seed=32)
    latents = [
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    ]
    calls = {"encode": 0}

    def fake_encode_jepa_latent(states, *, eval_mode=False):
      value = latents[calls["encode"]]
      calls["encode"] += 1
      return value

    agent.online_network.encode_jepa_latent = fake_encode_jepa_latent

    loss = agent._temporal_straightening_loss({
        "state": torch.zeros((1, 3, 4, 84, 84), dtype=torch.uint8),
        "same_trajectory": torch.ones((1, 3), dtype=torch.uint8),
    })

    assert calls["encode"] == 3
    assert loss.item() == pytest.approx(1.0)


def test_temporal_straightening_subsequence_len_is_enabled_only_when_weighted():
    from src.algorithms.atari100k.algorithm import Atari100KAlgorithm
    from src.algorithms.atari100k.der import DERConfig

    algorithm = Atari100KAlgorithm(device=torch.device("cpu"), num_actions=4)

    assert algorithm._subseq_len(DERConfig(num_actions=4)) == 1
    assert algorithm._subseq_len(
        DERConfig(
            num_actions=4,
            transfer_mode="jepa_full_finetune",
            temporal_straightening_weight=0.1,
        )
    ) == 3


def test_sigreg_loss_is_zero_when_disabled():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dqn",
        transfer_mode="jepa_full_finetune",
        lambda_sigreg=0.0,
        hidden_dim=4,
        device="cpu",
    )
    agent = DERAgent(config, seed=33)

    loss = agent._sigreg_loss(
        torch.zeros((2, 4, 84, 84), dtype=torch.uint8),
        torch.zeros((2, 4, 84, 84), dtype=torch.uint8),
    )

    assert loss.item() == pytest.approx(0.0)


def test_sigreg_embedding_loss_is_deterministic_and_finite():
    from src.algorithms.atari100k.der import DERAgent, DERConfig

    config = DERConfig(
        num_actions=4,
        encoder_type="dqn",
        transfer_mode="jepa_full_finetune",
        lambda_sigreg=0.1,
        hidden_dim=4,
        device="cpu",
    )
    agent = DERAgent(config, seed=34)
    embeddings = torch.randn((8, 4), generator=torch.Generator().manual_seed(1))

    loss = agent._sigreg_embedding_loss(embeddings)
    repeated_loss = agent._sigreg_embedding_loss(embeddings)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert repeated_loss.item() == pytest.approx(loss.item())


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
    lrs = optimizer_lrs_by_parameter_name(agent)
    assert lrs["encoder.input_adapter.weight"] == 1e-4
    assert lrs["encoder.input_adapter.bias"] == 1e-4
    assert lrs["encoder.reducer.weight"] == 1e-4
    assert lrs["encoder.reducer.bias"] == 1e-4
    assert lrs["encoder.layers.0.0.conv1.weight"] == 1e-5


def test_attentive_probe_uses_attention_projection_and_freezes_encoder():
    from src.algorithms.atari100k.der import DERAgent, DERConfig
    from src.algorithms.atari100k.transfer_learning import AttentiveProbe

    config = DERConfig(
        num_actions=4,
        encoder_type="resnet18",
        transfer_mode="attentive_probe",
        hidden_dim=128,
        device="cpu",
    )
    agent = DERAgent(config, seed=11)

    assert isinstance(agent.online_network.projection, AttentiveProbe)
    trainable_encoder_params = [
        name
        for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_encoder_params == [
        "input_adapter.weight",
        "input_adapter.bias",
        "reducer.weight",
        "reducer.bias",
    ]
    assert agent.online_network.latent_dim == 64
    assert agent.online_network.projection.value.in_features == 2304
    assert agent.online_network.projection.value.out_features == 128
    output = agent.online_network(
        torch.randint(0, 256, (2, 84, 84, 4), dtype=torch.uint8),
        agent.support,
    )
    assert output.q_values.shape == (2, 4)


def test_lora_mode_trains_only_encoder_adapters_and_heads():
    from src.algorithms.atari100k.der import DERAgent, DERConfig
    from src.algorithms.atari100k.transfer_learning import LoRAConv2d

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


def test_sac_bbf_lora_reset_keeps_adapter_state_dict_compatible():
    from src.algorithms.atari100k.sac_bbf import SACBBFAgent, SACBBFConfig

    config = SACBBFConfig(
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
    agent = SACBBFAgent(config, seed=29)

    agent.training_steps = 3
    agent.reset_weights()

    trainable_encoder_names = [
        name for name, parameter in agent.online_network.encoder.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_encoder_names
    assert all(".lora_" in name for name in trainable_encoder_names)


def test_sac_bbf_keeps_adapter_and_reducer_on_base_lr():
    from src.algorithms.atari100k.sac_bbf import SACBBFAgent, SACBBFConfig

    config = SACBBFConfig(
        num_actions=4,
        batch_size=2,
        encoder_type="resnet18",
        transfer_mode="full_finetune",
        encoder_lr_scale=0.1,
        hidden_dim=128,
        policy_learning_rate=3e-4,
        target_update_period=1,
        device="cpu",
    )
    agent = SACBBFAgent(config, seed=31)
    lrs = optimizer_lrs_by_parameter_name(agent)

    assert lrs["encoder.input_adapter.weight"] == 1e-4
    assert lrs["encoder.input_adapter.bias"] == 1e-4
    assert lrs["encoder.reducer.weight"] == 1e-4
    assert lrs["encoder.reducer.bias"] == 1e-4
    assert lrs["encoder.layers.0.0.conv1.weight"] == 1e-5
    assert lrs["policy.weight"] == 3e-4


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


def test_subsequence_replay_uses_horizon_aligned_next_state_and_terminal():
    from src.algorithms.atari100k.replay import SubsequenceReplayBuffer

    replay = SubsequenceReplayBuffer(
        observation_shape=(1,),
        stack_size=1,
        replay_capacity=20,
        batch_size=1,
        subseq_len=1,
        update_horizon=3,
        gamma=0.5,
        seed=0,
    )
    for i in range(8):
        replay.add(
            np.array([[i]], dtype=np.uint8),
            np.array([0], dtype=np.int32),
            np.array([float(i)], dtype=np.float32),
            np.array([1 if i == 4 else 0], dtype=np.uint8),
        )

    batch = replay.sample_transition_batch(
        indices=(np.array([2]), np.array([0]), np.array([0])),
    )

    assert batch["state"][0, 0, 0, 0] == 2
    assert batch["return"][0, 0] == pytest.approx(2.0 + 0.5 * 3.0 + 0.25 * 4.0)
    assert batch["discount"][0, 0] == pytest.approx(0.5**3)
    assert batch["terminal"][0, 0]
    assert batch["next_state"][0, 0, 0, 0] == 5
