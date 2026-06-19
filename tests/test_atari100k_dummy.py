"""Coverage for the constant-action Atari script-test algorithm."""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from tests.conftest import load_experiment_cfg


@pytest.mark.parametrize(
    "experiment",
    ["atari100k/dummy/qbert", "atari100k/dummy/battlezone"],
)
def test_dummy_atari_experiment_configs_compose(experiment: str):
    cfg = load_experiment_cfg(
        experiment,
        ["logger=[]", "trainer.accelerator=cpu", "trainer.devices=[0]"],
    )

    assert cfg.algorithm._target_.endswith("DummyAtari100KAlgorithm")
    assert cfg.trainer.total_frames == 20
    assert cfg.trainer.eval_every_n_steps == 10
    assert cfg.trainer.num_eval_episodes == 1
    assert cfg.eval_environment.transforms[-1].max_steps == 100


def test_constant_zero_policy_emits_one_hot_action_zero():
    from src.algorithms.atari100k.dummy import ConstantZeroPolicy

    policy = ConstantZeroPolicy(
        num_actions=4,
        one_hot_actions=True,
        dtype=torch.int64,
        device=torch.device("cpu"),
    )
    td = TensorDict({}, batch_size=[3])

    result = policy(td)

    torch.testing.assert_close(
        result["action"],
        torch.tensor(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            dtype=torch.int64,
        ),
    )


def test_smoke_dummy_atari_qbert_runs_training_and_evaluation(tmp_path):
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg(
        "atari100k/dummy/qbert",
        [
            "logger=[]",
            "trainer.accelerator=cpu",
            "trainer.devices=[0]",
            "trainer.total_frames=2",
            "trainer.log_every_n_steps=1",
            "trainer.eval_every_n_steps=1",
            "trainer.num_eval_episodes=1",
            "trainer.final_num_eval_episodes=1",
            "eval_environment.transforms.4.max_steps=10",
            f"checkpoint.save_dir={tmp_path}/checkpoints",
            "checkpoint.save_every_n_steps=1",
            "checkpoint.save_last=true",
            f"hydra.run.dir={tmp_path}",
        ],
    )
    from src.train import _train

    metrics = _train(cfg)

    assert metrics["train/dummy_action"] == 0.0
    assert metrics["train/processed_frames"] == 2.0
