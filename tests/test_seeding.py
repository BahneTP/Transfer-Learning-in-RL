"""Tests for process and environment reproducibility."""
from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch

from src.environments.atari_wrappers import NoopResetEnv
from src.environments.environment import Environment
from src.utils.seeding import derive_seed, seed_everything


def test_seed_everything_repeats_process_rngs():
    seed_everything(123, deterministic=True)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    seed_everything(123, deterministic=True)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    assert torch.are_deterministic_algorithms_enabled()


def test_derive_seed_is_stable_and_separates_streams():
    assert derive_seed(42, "train_environment") == derive_seed(
        42, "train_environment"
    )
    assert derive_seed(42, "train_environment") != derive_seed(
        42, "eval_environment"
    )


def test_environment_seed_repeats_initial_observation():
    environment = Environment(name="CartPole-v1")
    first_env = environment.make_env(seed=123)
    second_env = environment.make_env(seed=123)
    try:
        first = first_env.reset()["observation"]
        second = second_env.reset()["observation"]
    finally:
        first_env.close()
        second_env.close()

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_parallel_environment_seed_repeats_worker_observations():
    environment = Environment(name="CartPole-v1")
    first_env = environment.make_env(num_envs=2, seed=321)
    second_env = environment.make_env(num_envs=2, seed=321)
    try:
        first = first_env.reset()["observation"]
        second = second_env.reset()["observation"]
    finally:
        first_env.close()
        second_env.close()

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first[0], first[1])


def test_noop_reset_uses_environment_seed():
    class CountingEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (1,), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def __init__(self) -> None:
            super().__init__()
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.array([0], dtype=np.uint8), {}

        def step(self, action):
            self.steps += 1
            return np.array([self.steps], dtype=np.uint8), 0.0, False, False, {}

    env = NoopResetEnv(CountingEnv(), noop_max=30)
    first_observation, _ = env.reset(seed=987)
    first_steps = env.unwrapped.steps
    second_observation, _ = env.reset(seed=987)

    assert first_steps == env.unwrapped.steps
    np.testing.assert_array_equal(first_observation, second_observation)
