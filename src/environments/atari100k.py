"""Reference-style Atari 100K Gymnasium environments.

This module keeps the Atari preprocessing local instead of relying on TorchRL's
generic Atari transform stack.  The wrapper order mirrors the standalone
BBF-pytorch runner: no-op reset, max-and-skip, optional episodic life, grayscale
resize, then four-frame stacking.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _unwrap_reset(result: Any) -> np.ndarray:
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def _unwrap_step(result: Any) -> tuple[np.ndarray, float, bool, dict]:
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), info
    observation, reward, done, info = result
    return observation, float(reward), bool(done), info


def _resize_to_84_grayscale(frame: np.ndarray) -> np.ndarray:
    try:
        import cv2

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        from PIL import Image

        image = Image.fromarray(frame)
        if image.mode != "L":
            image = image.convert("L")
        image = image.resize((84, 84), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


class _NoopResetEnv:
    def __init__(self, env: Any, noop_max: int, rng: np.random.Generator) -> None:
        self.env = env
        self.noop_max = int(noop_max)
        self.rng = rng
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs: Any) -> np.ndarray:
        obs = _unwrap_reset(self.env.reset(**kwargs))
        noops = int(self.rng.integers(1, self.noop_max + 1)) if self.noop_max > 0 else 0
        for _ in range(noops):
            obs, _, done, _ = _unwrap_step(self.env.step(0))
            if done:
                obs = _unwrap_reset(self.env.reset(**kwargs))
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        return _unwrap_step(self.env.step(action))

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class _MaxAndSkipEnv:
    def __init__(self, env: Any, skip: int) -> None:
        self.env = env
        self.skip = int(skip)
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=2)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs: Any) -> np.ndarray:
        self._obs_buffer.clear()
        return _unwrap_reset(self.env.reset(**kwargs))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        total_reward = 0.0
        done = False
        info: dict = {}
        obs = None
        for _ in range(self.skip):
            obs, reward, done, info = _unwrap_step(self.env.step(action))
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        if len(self._obs_buffer) == 2:
            obs = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return obs, total_reward, done, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class _EpisodicLifeEnv:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.lives = 0
        self.was_real_done = True
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _lives(self) -> int:
        ale = getattr(getattr(self.env, "unwrapped", self.env), "ale", None)
        return int(ale.lives()) if ale is not None else 0

    def reset(self, **kwargs: Any) -> np.ndarray:
        if self.was_real_done:
            obs = _unwrap_reset(self.env.reset(**kwargs))
        else:
            obs, _, _, _ = _unwrap_step(self.env.step(0))
        self.lives = self._lives()
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = _unwrap_step(self.env.step(action))
        self.was_real_done = done
        lives = self._lives()
        info = dict(info)
        info["real_done"] = done
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class _AtariPreprocessEnv:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs: Any) -> np.ndarray:
        return _resize_to_84_grayscale(_unwrap_reset(self.env.reset(**kwargs)))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = _unwrap_step(self.env.step(action))
        return _resize_to_84_grayscale(obs), reward, done, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class Atari100KEnv:
    """Gymnasium-compatible Atari 100K env returning ``{"pixels": CHW uint8}``."""

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        game: str,
        *,
        seed: int | None = None,
        noop_max: int = 30,
        frame_skip: int = 4,
        stack_size: int = 4,
        terminal_on_life_loss: bool = True,
    ) -> None:
        import ale_py
        import gymnasium as gym
        from gymnasium import spaces

        if hasattr(gym, "register_envs"):
            gym.register_envs(ale_py)

        env_id = game if "/" in game or game.endswith("-v4") or game.endswith("-v5") else f"ALE/{game}-v5"
        self.env = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=0.0,
            disable_env_checker=True,
        )
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            self.env.reset(seed=seed)
        self.env = _NoopResetEnv(self.env, noop_max=noop_max, rng=self.rng)
        self.env = _MaxAndSkipEnv(self.env, skip=frame_skip)
        if terminal_on_life_loss:
            self.env = _EpisodicLifeEnv(self.env)
        self.env = _AtariPreprocessEnv(self.env)
        self.action_space = self.env.action_space
        self.stack_size = int(stack_size)
        self.frames: deque[np.ndarray] = deque(maxlen=self.stack_size)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.stack_size, 84, 84),
                    dtype=np.uint8,
                )
            }
        )

    @property
    def unwrapped(self) -> "Atari100KEnv":
        return self

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        del options
        kwargs = {"seed": seed} if seed is not None else {}
        obs = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return {"pixels": self._stacked_obs()}, {}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        obs, reward, done, info = self.env.step(int(action))
        self.frames.append(obs)
        return {"pixels": self._stacked_obs()}, reward, bool(done), False, info

    def close(self) -> None:
        self.env.close()

    def _stacked_obs(self) -> np.ndarray:
        return np.stack(list(self.frames), axis=0).astype(np.uint8)
