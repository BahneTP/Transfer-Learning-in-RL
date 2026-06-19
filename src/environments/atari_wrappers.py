"""Gymnasium Atari wrappers matching the standalone BBF-pytorch runner."""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    """Start real games from a randomized state using 1..noop_max NOOPs."""

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self._rng = np.random.default_rng()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        noops = (
            int(self._rng.integers(1, self.noop_max + 1))
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(options=options)
        return observation, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat actions and max-pool the final two raw Atari frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self.skip = skip
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=2)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._obs_buffer.clear()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        observation = None
        terminated = truncated = False
        info: dict[str, Any] = {}
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(observation)
            total_reward += float(reward)
            if terminated or truncated:
                break
        if len(self._obs_buffer) == 2:
            observation = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return observation, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Expose life loss as terminal while preserving the underlying game."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def _lives(self) -> int:
        ale = getattr(self.unwrapped, "ale", None)
        return int(ale.lives()) if ale is not None else 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.was_real_done:
            observation, info = self.env.reset(seed=seed, options=options)
        else:
            # Match BBF-pytorch: advance from the life-loss screen with one
            # NOOP agent action (including the inner action repeat), without
            # resetting the actual ALE game.
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(seed=seed, options=options)
        self.lives = self._lives()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = bool(terminated or truncated)
        lives = self._lives()
        life_lost = lives < self.lives and lives > 0
        self.lives = lives
        if life_lost:
            terminated = True
        return observation, reward, terminated, truncated, info


def wrap_atari(
    env: gym.Env,
    *,
    noop_max: int = 30,
    frame_skip: int = 4,
    terminal_on_life_loss: bool,
) -> gym.Env:
    """Apply the BBF-pytorch Atari wrapper order."""
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if terminal_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env
