"""Rainbow for Atari 100K."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.nn import functional as F
from torchrl.envs import EnvBase

from src.algorithms.atari100k.networks import RainbowDQNNetwork
from src.algorithms.atari100k.replay import PrioritizedNStepReplay
from src.algorithms.atari100k.rl import categorical_target
from src.algorithms.atari100k.rl import linearly_decaying_epsilon
from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState


class _RainbowPolicy(nn.Module):
    def __init__(self, algorithm: "RainbowAtari100KAlgorithm", *, eval_mode: bool) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.eval_mode = eval_mode

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            network = self.algorithm.online_network
            noisy_eval_mode = self.eval_mode and not self.algorithm.eval_noise
            network.eval() if self.eval_mode else network.train()
            output = network(
                observation.to(self.algorithm.device),
                self.algorithm.support,
                eval_mode=noisy_eval_mode,
            )
            q_values = output.q_values
            if self.eval_mode:
                epsilon = self.algorithm.epsilon_eval
            else:
                epsilon = linearly_decaying_epsilon(
                    self.algorithm.epsilon_decay_period,
                    self.algorithm._collected_frames,
                    self.algorithm.min_replay_history,
                    self.algorithm.epsilon_train,
                )
            greedy_actions = q_values.argmax(dim=-1)
            random_actions = torch.randint(
                0,
                self.algorithm.num_actions,
                greedy_actions.shape,
                device=self.algorithm.device,
                generator=self.algorithm.generator,
            )
            draw = torch.rand(greedy_actions.shape, device=self.algorithm.device, generator=self.algorithm.generator)
            actions = torch.where(draw <= epsilon, random_actions, greedy_actions)
        return q_values, actions


class RainbowAtari100KAlgorithm(BaseAlgorithm):
    """Rainbow with C51, Double DQN, noisy/dueling heads and prioritized n-step replay."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "pixels",
        replay_capacity: int = 200_000,
        observation_shape: tuple[int, int, int] = (4, 84, 84),
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        gamma: float = 0.99,
        update_horizon: int = 10,
        min_replay_history: int = 1600,
        target_update_period: int = 8000,
        epsilon_train: float = 0.01,
        epsilon_eval: float = 0.001,
        epsilon_decay_period: int = 2000,
        replay_ratio: int = 32,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        adam_eps: float = 1.5e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 10.0,
        noisy: bool = True,
        dueling: bool = True,
        double_dqn: bool = True,
        eval_noise: bool = True,
        target_eval_mode: bool = False,
        target_update_tau: float = 1.0,
        frames_per_batch: int = 1,
        max_frames_per_traj: int = -1,
        seed: int = 1,
    ) -> None:
        super().__init__(device)
        self.obs_key = obs_key
        self.replay_capacity = replay_capacity
        self.observation_shape = observation_shape
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.replay_ratio = replay_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.noisy = noisy
        self.dueling = dueling
        self.double_dqn = double_dqn
        self.eval_noise = eval_noise
        self.target_eval_mode = target_eval_mode
        self.target_update_tau = target_update_tau
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self.seed = seed
        self._collected_frames = 0
        self.gradient_steps = 0

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        self.num_actions = int(proof_env.action_spec.space.n)
        proof_env.close()
        self.device = self.device or torch.device("cpu")
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        self.online_network = RainbowDQNNetwork(
            num_actions=self.num_actions,
            num_atoms=self.num_atoms,
            noisy=self.noisy,
            dueling=self.dueling,
            distributional=True,
            encoder_type="dqn",
            hidden_dim=512,
            width_scale=1,
            input_channels=self.observation_shape[0],
        ).to(self.device)
        self.target_network = RainbowDQNNetwork(
            num_actions=self.num_actions,
            num_atoms=self.num_atoms,
            noisy=self.noisy,
            dueling=self.dueling,
            distributional=True,
            encoder_type="dqn",
            hidden_dim=512,
            width_scale=1,
            input_channels=self.observation_shape[0],
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.AdamW(
            self._optimizer_groups(),
            lr=self.learning_rate,
            eps=self.adam_eps,
        )
        self.replay = PrioritizedNStepReplay(
            capacity=self.replay_capacity,
            observation_shape=self.observation_shape,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            seed=self.seed,
        )
        self._policy = TensorDictModule(
            _RainbowPolicy(self, eval_mode=True),
            in_keys=[self.obs_key],
            out_keys=["action_value", "action"],
        )
        self._explore_policy = TensorDictModule(
            _RainbowPolicy(self, eval_mode=False),
            in_keys=[self.obs_key],
            out_keys=["action_value", "action"],
        )

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.min_replay_history,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    def step(self, batch: TensorDict) -> dict[str, float]:
        flat = batch.reshape(-1)
        self._store_batch(flat)
        self._collected_frames += flat.numel()
        epsilon = linearly_decaying_epsilon(
            self.epsilon_decay_period,
            self._collected_frames,
            self.min_replay_history,
            self.epsilon_train,
        )
        min_sample_history = max(
            self.min_replay_history,
            self.replay.stack_size + self.replay.update_horizon + 1,
        )
        if len(self.replay) < min_sample_history:
            return {"train/epsilon": float(epsilon), "train/replay_size": float(len(self.replay))}

        updates = max(1, int(self.replay_ratio * flat.numel() // self.batch_size))
        losses = torch.zeros(updates, device=self.device)
        grad_norms = torch.zeros(updates, device=self.device)
        for update in range(updates):
            metrics = self._train_one_update()
            losses[update] = metrics["loss"]
            grad_norms[update] = metrics["grad_norm"]
        return {
            "train/q_loss": losses.mean().item(),
            "train/grad_norm": grad_norms.mean().item(),
            "train/epsilon": float(epsilon),
            "train/replay_size": float(len(self.replay)),
        }

    def get_policy(self) -> TensorDictModule:
        return self._policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    def _train_one_update(self) -> dict[str, torch.Tensor]:
        sample = self.replay.sample(self.batch_size, self.device)
        with torch.no_grad():
            next_online = self.online_network(sample.next_states, self.support, eval_mode=False)
            next_target = self.target_network(
                sample.next_states,
                self.support,
                eval_mode=self.target_eval_mode,
            )
            target = categorical_target(
                sample.returns,
                sample.terminals,
                self.support,
                sample.discounts,
                next_online.q_values,
                next_target.q_values,
                next_target.probabilities,
                double_dqn=self.double_dqn,
            )
        output = self.online_network(sample.states, self.support, eval_mode=False)
        chosen_logits = output.logits[torch.arange(sample.actions.shape[0], device=self.device), sample.actions]
        per_sample_loss = -(target * F.log_softmax(chosen_logits, dim=-1)).sum(dim=-1)
        loss = (sample.weights * per_sample_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._maybe_update_target()
        priorities = torch.sqrt(per_sample_loss.detach() + 1e-10).cpu().numpy()
        self.replay.set_priority(sample.indices, priorities)
        self.gradient_steps += 1
        return {"loss": loss.detach(), "grad_norm": torch.as_tensor(grad_norm, device=self.device)}

    def _store_batch(self, batch: TensorDict) -> None:
        states = batch.get(self.obs_key)
        next_states = batch.get(("next", self.obs_key))
        actions = batch.get("action")
        rewards = batch.get(("next", "reward")).reshape(-1)
        terminals = batch.get(("next", "end-of-life"), default=None)
        if terminals is None:
            terminals = batch.get(("next", "done"))
        terminals = terminals.reshape(-1)
        if actions.ndim > 1 and actions.shape[-1] == self.num_actions:
            actions = actions.argmax(dim=-1)
        actions = actions.reshape(-1)
        for i in range(batch.numel()):
            self.replay.add(
                states[i],
                int(actions[i].item()),
                float(rewards[i].item()),
                bool(terminals[i].item()),
                next_states[i],
            )

    def _maybe_update_target(self) -> None:
        if self.gradient_steps % self.target_update_period != 0:
            return
        if self.target_update_tau >= 1.0:
            self.target_network.load_state_dict(self.online_network.state_dict())
            return
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.online_network.parameters(),
            ):
                target_param.mul_(1.0 - self.target_update_tau)
                target_param.add_(online_param, alpha=self.target_update_tau)

    def _optimizer_groups(self) -> list[dict]:
        decay_params = []
        no_decay_params = []
        for parameter in self.online_network.parameters():
            if parameter.ndim == 1:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)
        return [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict={
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
            },
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={
                "replay": self.replay.state_dict(),
                "collected_frames": self._collected_frames,
                "gradient_steps": self.gradient_steps,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.online_network.load_state_dict(state.policy_state_dict["online_network"])
        self.target_network.load_state_dict(state.policy_state_dict["target_network"])
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        if state.extra:
            self.replay.load_state_dict(state.extra["replay"])
            self._collected_frames = int(state.extra["collected_frames"])
            self.gradient_steps = int(state.extra["gradient_steps"])
