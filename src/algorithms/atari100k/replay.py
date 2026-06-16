"""Prioritized n-step replay for Atari 100K algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.depth = int(math.ceil(math.log2(capacity)))
        self.low_idx = (2**self.depth) - 1
        self.nodes = np.zeros(2 ** (self.depth + 1) - 1, dtype=np.float64)
        self.highest_set = 0
        self.max_recorded_priority = 1.0

    @property
    def total_priority(self) -> float:
        return float(self.nodes[0])

    def set(self, index: int, value: float) -> None:
        value = float(max(value, 0.0))
        self.highest_set = max(int(index), self.highest_set)
        self.max_recorded_priority = max(value, self.max_recorded_priority)
        tree_index = index + self.low_idx
        delta = value - self.nodes[tree_index]
        while True:
            self.nodes[tree_index] += delta
            if tree_index == 0:
                break
            tree_index = (tree_index - 1) // 2

    def get(self, indices: np.ndarray) -> np.ndarray:
        return self.nodes[indices + self.low_idx]

    def stratified_sample(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if self.total_priority <= 0.0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        queries = (np.arange(batch_size, dtype=np.float64) + rng.random(batch_size)) / batch_size
        scaled = queries * self.total_priority
        indices = np.zeros(batch_size, dtype=np.int64)
        for _ in range(self.depth):
            left = indices * 2 + 1
            left_sum = self.nodes[left]
            go_right = scaled >= left_sum
            scaled = np.where(go_right, scaled - left_sum, scaled)
            indices = np.where(go_right, left + 1, left)
        return np.minimum(indices - self.low_idx, self.highest_set).astype(np.int64)


@dataclass
class ReplayBatch:
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    terminals: torch.Tensor
    discounts: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedNStepReplay:
    """Dopamine-style prioritized replay that stores single Atari frames."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_shape: tuple[int, ...],
        update_horizon: int,
        gamma: float,
        seed: int,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)
        if len(self.observation_shape) != 3:
            raise ValueError(f"Expected CHW observation_shape, got {self.observation_shape}")
        self.stack_size = int(self.observation_shape[0])
        self.frame_shape = tuple(self.observation_shape[1:])
        self.update_horizon = int(update_horizon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.observations = np.zeros((capacity, *self.frame_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.episode_ends = np.zeros((capacity,), dtype=np.bool_)
        self.cursor = 0
        self.size = 0
        self.total_added = 0
        self.sum_tree = SumTree(capacity)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        next_state: torch.Tensor,
    ) -> None:
        index = self.cursor
        self.observations[index] = self._latest_frame_uint8(state)
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = bool(done)
        self.episode_ends[index] = bool(done)
        self.sum_tree.set(index, self.sum_tree.max_recorded_priority)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        indices = self._sample_indices(batch_size)
        returns = np.zeros((batch_size,), dtype=np.float32)
        discounts = np.zeros((batch_size,), dtype=np.float32)
        terminals = np.zeros((batch_size,), dtype=np.float32)
        bootstrap_indices = np.zeros((batch_size,), dtype=np.int64)
        for row, index in enumerate(indices):
            ret = 0.0
            discount = 1.0
            terminal = False
            for n in range(self.update_horizon):
                current = (index + n) % self.capacity
                ret += discount * float(self.rewards[current])
                # Match the Dopamine-style replay convention: terminal masking
                # covers the first update_horizon - 1 bootstrap transitions.
                terminal = bool(self.dones[current]) if n < self.update_horizon - 1 else False
                if terminal:
                    break
                discount *= self.gamma
            returns[row] = ret
            discounts[row] = discount
            terminals[row] = float(terminal)
            bootstrap_indices[row] = (index + self.update_horizon - 1) % self.capacity

        probabilities = self.sum_tree.get(indices)
        weights = (1.0 / np.sqrt(probabilities + 1e-10)).astype(np.float32)
        weights /= max(float(weights.max()), 1e-10)

        return ReplayBatch(
            states=torch.as_tensor(self._stack_batch(indices), device=device),
            next_states=torch.as_tensor(self._stack_batch(bootstrap_indices), device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            returns=torch.as_tensor(returns, device=device),
            terminals=torch.as_tensor(terminals, device=device),
            discounts=torch.as_tensor(discounts, device=device),
            weights=torch.as_tensor(weights, device=device),
            indices=indices,
        )

    def set_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(int(index), float(priority))

    def state_dict(self) -> dict:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "episode_ends": self.episode_ends,
            "cursor": self.cursor,
            "size": self.size,
            "total_added": self.total_added,
            "sum_tree_nodes": self.sum_tree.nodes,
            "sum_tree_highest_set": self.sum_tree.highest_set,
            "sum_tree_max": self.sum_tree.max_recorded_priority,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.observations[:] = state["observations"]
        self.actions[:] = state["actions"]
        self.rewards[:] = state["rewards"]
        self.dones[:] = state["dones"]
        self.episode_ends[:] = state.get("episode_ends", state["dones"])
        self.cursor = int(state["cursor"])
        self.size = int(state["size"])
        self.total_added = int(state["total_added"])
        self.sum_tree.nodes[:] = state["sum_tree_nodes"]
        self.sum_tree.highest_set = int(state.get("sum_tree_highest_set", max(self.size - 1, 0)))
        self.sum_tree.max_recorded_priority = float(state["sum_tree_max"])
        self.rng.bit_generator.state = state["rng_state"]

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if self.size <= self.stack_size + self.update_horizon:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = np.zeros((batch_size,), dtype=np.int64)
        allowed_attempts = 1000
        for row in range(batch_size):
            index = int(self.sum_tree.stratified_sample(1, self.rng)[0])
            while not self._is_valid_index(index):
                allowed_attempts -= 1
                if allowed_attempts <= 0:
                    raise RuntimeError("Could not sample enough valid Atari replay transitions.")
                index = int(self.sum_tree.stratified_sample(1, self.rng)[0])
            indices[row] = index
        return indices

    def _is_valid_index(self, index: int) -> bool:
        if index < 0 or index >= self.capacity:
            return False
        if self.size < self.capacity:
            if index < self.stack_size - 1:
                return False
            if index >= self.cursor - self.update_horizon:
                return False
        invalid = {
            (self.cursor - self.update_horizon + i) % self.capacity
            for i in range(self.stack_size + self.update_horizon)
        }
        if index in invalid:
            return False
        for offset in range(self.update_horizon):
            current = (index + offset) % self.capacity
            if self.episode_ends[current] and not self.dones[current]:
                return False
        return True

    def _stack_batch(self, indices: np.ndarray) -> np.ndarray:
        return np.stack([self._stack_frames(int(index)) for index in indices], axis=0)

    def _stack_frames(self, index: int) -> np.ndarray:
        stack = np.zeros(self.observation_shape, dtype=np.uint8)
        first_valid = self._first_valid_stack_index(index)
        for slot, frame_index in enumerate(range(index - self.stack_size + 1, index + 1)):
            if frame_index >= first_valid:
                stack[slot] = self.observations[frame_index % self.capacity]
        return stack

    def _first_valid_stack_index(self, index: int) -> int:
        first_valid = index - self.stack_size + 1
        for offset in range(self.stack_size - 1):
            previous = index - self.stack_size + 1 + offset
            if self.dones[previous % self.capacity] or self.episode_ends[previous % self.capacity]:
                first_valid = previous + 1
        return first_valid

    def _latest_frame_uint8(self, state: torch.Tensor) -> np.ndarray:
        state = state.detach().cpu()
        scale_from_unit_float = torch.is_floating_point(state)
        if state.ndim == 4 and state.shape[0] == 1:
            state = state[0]
        if state.shape[-1] in (1, 3, 4) and state.shape[0] not in (1, 3, 4):
            state = state.permute(2, 0, 1)
        if state.ndim == 3:
            state = state[-1]
        state = state.float()
        if scale_from_unit_float and state.max() <= 1.5:
            state = state * 255.0
        return state.clamp(0, 255).to(torch.uint8).numpy()
