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
        self.max_recorded_priority = 1.0

    @property
    def total_priority(self) -> float:
        return float(self.nodes[0])

    def set(self, index: int, value: float) -> None:
        value = float(max(value, 0.0))
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
        return np.minimum(indices - self.low_idx, self.capacity - 1).astype(np.int64)


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
    future_states: torch.Tensor | None = None
    rollout_actions: torch.Tensor | None = None
    same_trajectory: torch.Tensor | None = None


class PrioritizedNStepReplay:
    """Stores stacked Atari transitions and computes n-step targets on sample."""

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
        self.update_horizon = int(update_horizon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, *self.observation_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *self.observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
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
        self.states[index] = self._to_uint8(state)
        self.next_states[index] = self._to_uint8(next_state)
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = bool(done)
        self.sum_tree.set(index, self.sum_tree.max_recorded_priority)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        indices = self._sample_indices(batch_size)
        return self._build_batch(indices, device, jumps=0)

    def sample_sequence(self, batch_size: int, device: torch.device, *, jumps: int) -> ReplayBatch:
        indices = self._sample_indices(batch_size)
        return self._build_batch(indices, device, jumps=jumps)

    def _build_batch(self, indices: np.ndarray, device: torch.device, *, jumps: int) -> ReplayBatch:
        batch_size = int(indices.shape[0])
        returns = np.zeros((batch_size,), dtype=np.float32)
        discounts = np.zeros((batch_size,), dtype=np.float32)
        terminals = np.zeros((batch_size,), dtype=np.float32)
        bootstrap_indices = np.zeros((batch_size,), dtype=np.int64)
        for row, index in enumerate(indices):
            ret = 0.0
            discount = 1.0
            last_index = index
            terminal = False
            for n in range(self.update_horizon):
                current = (index + n) % self.capacity
                ret += discount * float(self.rewards[current])
                last_index = current
                terminal = bool(self.dones[current])
                if terminal:
                    break
                discount *= self.gamma
            returns[row] = ret
            discounts[row] = discount
            terminals[row] = float(terminal)
            bootstrap_indices[row] = last_index

        probabilities = self.sum_tree.get(indices) / max(self.sum_tree.total_priority, 1e-12)
        weights = (1.0 / np.sqrt(probabilities + 1e-10)).astype(np.float32)
        weights /= max(float(weights.max()), 1e-10)

        return ReplayBatch(
            states=torch.as_tensor(self.states[indices], device=device),
            next_states=torch.as_tensor(self.next_states[bootstrap_indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            returns=torch.as_tensor(returns, device=device),
            terminals=torch.as_tensor(terminals, device=device),
            discounts=torch.as_tensor(discounts, device=device),
            weights=torch.as_tensor(weights, device=device),
            indices=indices,
            future_states=self._future_states(indices, jumps, device) if jumps > 0 else None,
            rollout_actions=self._rollout_actions(indices, jumps, device) if jumps > 0 else None,
            same_trajectory=self._same_trajectory(indices, jumps, device) if jumps > 0 else None,
        )

    def set_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(int(index), float(priority) ** self.alpha)

    def state_dict(self) -> dict:
        return {
            "states": self.states,
            "next_states": self.next_states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "cursor": self.cursor,
            "size": self.size,
            "total_added": self.total_added,
            "sum_tree_nodes": self.sum_tree.nodes,
            "sum_tree_max": self.sum_tree.max_recorded_priority,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.states[:] = state["states"]
        self.next_states[:] = state["next_states"]
        self.actions[:] = state["actions"]
        self.rewards[:] = state["rewards"]
        self.dones[:] = state["dones"]
        self.cursor = int(state["cursor"])
        self.size = int(state["size"])
        self.total_added = int(state["total_added"])
        self.sum_tree.nodes[:] = state["sum_tree_nodes"]
        self.sum_tree.max_recorded_priority = float(state["sum_tree_max"])
        self.rng.bit_generator.state = state["rng_state"]

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = self.sum_tree.stratified_sample(batch_size, self.rng)
        if self.size < self.capacity:
            indices = np.minimum(indices, self.size - 1)
        return indices

    def _future_states(self, indices: np.ndarray, jumps: int, device: torch.device) -> torch.Tensor:
        future = np.zeros((indices.shape[0], jumps, *self.observation_shape), dtype=np.uint8)
        for row, index in enumerate(indices):
            for j in range(jumps):
                future[row, j] = self.next_states[(index + j) % self.capacity]
        return torch.as_tensor(future, device=device)

    def _rollout_actions(self, indices: np.ndarray, jumps: int, device: torch.device) -> torch.Tensor:
        actions = np.zeros((indices.shape[0], jumps), dtype=np.int64)
        for row, index in enumerate(indices):
            for j in range(jumps):
                actions[row, j] = self.actions[(index + j) % self.capacity]
        return torch.as_tensor(actions, device=device)

    def _same_trajectory(self, indices: np.ndarray, jumps: int, device: torch.device) -> torch.Tensor:
        mask = np.ones((indices.shape[0], jumps), dtype=np.float32)
        for row, index in enumerate(indices):
            alive = True
            for j in range(jumps):
                mask[row, j] = 1.0 if alive else 0.0
                if self.dones[(index + j) % self.capacity]:
                    alive = False
        return torch.as_tensor(mask, device=device)

    def _to_uint8(self, state: torch.Tensor) -> np.ndarray:
        state = state.detach().cpu()
        if state.ndim == 4 and state.shape[0] == 1:
            state = state[0]
        if state.shape[-1] in (1, 3, 4) and state.shape[0] not in (1, 3, 4):
            state = state.permute(2, 0, 1)
        state = state.float()
        if state.max() <= 1.5:
            state = state * 255.0
        return state.clamp(0, 255).to(torch.uint8).numpy()
