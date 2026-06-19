"""Discrete SAC-BBF agent core in PyTorch."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.algorithms.atari100k.bbf import BBFAgent
from src.algorithms.atari100k.bbf import BBFConfig
from src.algorithms.atari100k.networks import RainbowDQNNetwork
from src.algorithms.atari100k.networks import SACRainbowDQNNetwork
from src.algorithms.atari100k.rl import project_distribution


@dataclasses.dataclass
class SACBBFConfig(BBFConfig):
  entropy_decay_period: int = 80_000
  entropy_initial_coef: float = 1e-2
  entropy_final_coef: float = 0.0
  policy_learning_rate: float = 1e-4
  alpha_learning_rate: float = 1e-3


class SACBBFAgent(BBFAgent):
  """SAC-BBF variant using a discrete policy head for acting and backups."""

  config: SACBBFConfig
  online_network: SACRainbowDQNNetwork
  target_network: SACRainbowDQNNetwork

  def _make_network(self) -> RainbowDQNNetwork:
    return SACRainbowDQNNetwork(
        num_actions=self.config.num_actions,
        num_atoms=self.config.num_atoms,
        noisy=self.config.noisy,
        dueling=self.config.dueling,
        distributional=self.config.distributional,
        encoder_type=self.config.encoder_type,  # type: ignore[arg-type]
        hidden_dim=self.config.hidden_dim,
        width_scale=self.config.width_scale,
        renormalize_output=self.config.renormalize_output,
        input_channels=self.config.stack_size,
    )

  def _make_optimizer(self) -> torch.optim.Optimizer:
    encoder_decay_params = []
    encoder_no_decay_params = []
    head_decay_params = []
    head_no_decay_params = []
    policy_decay_params = []
    policy_no_decay_params = []
    alpha_params = []
    for name, parameter in self.online_network.named_parameters():
      if not parameter.requires_grad:
        continue
      if name == "_log_alpha":
        alpha_params.append(parameter)
        continue
      has_weight_decay = parameter.ndim != 1
      if name.startswith(("encoder.", "transition_model.")):
        (encoder_decay_params if has_weight_decay else encoder_no_decay_params).append(parameter)
      elif name.startswith(("policy_projection", "predict_policy", "policy")):
        (policy_decay_params if has_weight_decay else policy_no_decay_params).append(parameter)
      else:
        (head_decay_params if has_weight_decay else head_no_decay_params).append(parameter)

    parameter_groups = []
    if encoder_decay_params:
      parameter_groups.append({
          "params": encoder_decay_params,
          "lr": self.config.learning_rate,
          "weight_decay": self.config.weight_decay,
      })
    if encoder_no_decay_params:
      parameter_groups.append({
          "params": encoder_no_decay_params,
          "lr": self.config.learning_rate,
          "weight_decay": 0.0,
      })
    if head_decay_params:
      parameter_groups.append({
          "params": head_decay_params,
          "lr": self.config.learning_rate,
          "weight_decay": self.config.weight_decay,
      })
    if head_no_decay_params:
      parameter_groups.append({
          "params": head_no_decay_params,
          "lr": self.config.learning_rate,
          "weight_decay": 0.0,
      })
    if policy_decay_params:
      parameter_groups.append({
          "params": policy_decay_params,
          "lr": self.config.policy_learning_rate,
          "weight_decay": self.config.weight_decay,
      })
    if policy_no_decay_params:
      parameter_groups.append({
          "params": policy_no_decay_params,
          "lr": self.config.policy_learning_rate,
          "weight_decay": 0.0,
      })
    if alpha_params:
      parameter_groups.append({
          "params": alpha_params,
          "lr": self.config.alpha_learning_rate,
          "weight_decay": 0.0,
      })
    return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999), eps=self.config.adam_eps)

  def _build_lazy_modules(self) -> None:
    height, width = self.config.observation_shape
    dummy = torch.zeros(
        (1, self.config.stack_size, height, width),
        dtype=torch.uint8,
        device=self.device,
    )
    dummy_actions = torch.zeros((1, max(self.config.jumps, 1)), dtype=torch.long, device=self.device)
    with torch.no_grad():
      self.online_network(dummy, self.support, actions=dummy_actions, do_rollout=True, eval_mode=True)
      self.target_network(dummy, self.support, actions=dummy_actions, do_rollout=True, eval_mode=True)
      self.online_network.get_policy(dummy, eval_mode=True)
      self.target_network.get_policy(dummy, eval_mode=True)
      self.online_network.encode_project(dummy, eval_mode=True)
      self.target_network.encode_project(dummy, eval_mode=True)

  def _build_specific_network(self, network: RainbowDQNNetwork) -> None:
    height, width = self.config.observation_shape
    dummy = torch.zeros(
        (1, self.config.stack_size, height, width),
        dtype=torch.uint8,
        device=self.device,
    )
    dummy_actions = torch.zeros((1, max(self.config.jumps, 1)), dtype=torch.long, device=self.device)
    with torch.no_grad():
      network(dummy, self.support, actions=dummy_actions, do_rollout=True, eval_mode=True)
      assert isinstance(network, SACRainbowDQNNetwork)
      network.get_policy(dummy, eval_mode=True)
      network.encode_project(dummy, eval_mode=True)

  def _keys_to_copy(self) -> tuple[str, ...]:
    return ("encoder", "transition_model", "_log_alpha")

  def train_step(self, batch: dict[str, Any]) -> dict[str, float | np.ndarray]:
    if self._should_reset():
      self.reset_weights()
    mini_batches = self._split_grouped_batch(batch)
    metrics = [self._train_one_minibatch(mini_batch) for mini_batch in mini_batches]
    return self._merge_group_metrics(metrics)

  def _train_one_minibatch(self, batch: dict[str, Any]) -> dict[str, float | np.ndarray]:
    self.online_network.train()
    states = self._batch_tensor(batch["state"])
    next_states = self._batch_tensor(batch["next_state"])
    actions = self._batch_tensor(batch["action"]).long()
    rewards = self._batch_tensor(batch["return"]).float()[:, 0]
    terminals = self._batch_tensor(batch["terminal"]).float()[:, 0]
    discounts = self._batch_tensor(batch["discount"]).float()[:, 0]
    same_trajectory = self._same_trajectory(batch, states)
    loss_weights = self._loss_weights(batch, states.shape[0])

    current_states = states[:, 0]
    backup_next_states = next_states[:, 0]
    current_actions = actions[:, 0]
    rollout_actions = actions[:, :-1]

    states_processed = None
    current_latent = None
    backup_next_processed = None
    future_processed = None
    if self.config.data_augmentation:
      states_processed = self._preprocess_state_sequence(states)
      current_latent = self.online_network.encode_processed(states_processed[:, 0])
      backup_next_processed = self._preprocess_state_batch(backup_next_states)
      if states.shape[1] > 1:
        future_processed = states_processed[:, 1:]

    with torch.no_grad():
      if backup_next_processed is None:
        next_target = self.target_network(
            backup_next_states,
            self.support,
            eval_mode=self.config.target_eval_mode,
            data_augmentation=self.config.data_augmentation,
        )
        next_policy_logits, next_actions = self.online_network.get_policy(
            backup_next_states,
            eval_mode=False,
            data_augmentation=self.config.data_augmentation,
        )
        del next_policy_logits
      else:
        next_online_latent = self.online_network.encode_processed(backup_next_processed)
        next_target_processed = (
            backup_next_processed
            if self.config.match_online_target_rngs
            else self._preprocess_state_batch(backup_next_states)
        )
        next_target_latent = self.target_network.encode_processed(next_target_processed)
        next_target = self.target_network.forward_from_latent(
            next_target_latent,
            self.support,
            eval_mode=self.config.target_eval_mode,
        )
        next_policy_logits = self.online_network.policy_logits_from_latent(
            next_online_latent,
            eval_mode=False,
        )
        next_actions = torch.distributions.Categorical(logits=next_policy_logits).sample()
      assert next_target.probabilities is not None
      next_probabilities = next_target.probabilities[
          torch.arange(next_actions.shape[0], device=self.device),
          next_actions,
      ]
      gamma_with_terminal = discounts * (1.0 - terminals.float())
      target_support = rewards[:, None] + gamma_with_terminal[:, None] * self.support[None, :]
      target = project_distribution(target_support, next_probabilities, self.support)
      spr_targets = self._spr_targets(states[:, 1:], future_processed=future_processed)

    if current_latent is None:
      output = self.online_network(
          current_states,
          self.support,
          actions=rollout_actions,
          do_rollout=self.config.spr_weight > 0 and rollout_actions.shape[1] > 0,
          eval_mode=False,
          data_augmentation=self.config.data_augmentation,
      )
      policy_logits, policy_samples = self.online_network.get_policy(
          current_states,
          eval_mode=False,
          data_augmentation=self.config.data_augmentation,
      )
    else:
      output = self.online_network.forward_from_latent(
          current_latent,
          self.support,
          actions=rollout_actions,
          do_rollout=self.config.spr_weight > 0 and rollout_actions.shape[1] > 0,
          eval_mode=False,
      )
      policy_logits = self.online_network.policy_logits_from_latent(current_latent, eval_mode=False)
      policy_samples = torch.distributions.Categorical(logits=policy_logits).sample()

    assert output.logits is not None
    chosen_logits = output.logits[
        torch.arange(current_actions.shape[0], device=self.device),
        current_actions,
    ]
    dqn_loss = -(target * F.log_softmax(chosen_logits, dim=-1)).sum(dim=-1)
    td_error = dqn_loss + torch.nan_to_num(
        target * torch.log(target.clamp_min(1e-8))
    ).sum(dim=-1)
    spr_loss = self._spr_loss(output.latent, spr_targets, same_trajectory)
    q_sac_loss = dqn_loss + self.config.spr_weight * spr_loss

    log_prob = F.log_softmax(policy_logits.float(), dim=-1)
    prob = F.softmax(policy_logits.float(), dim=-1)
    sampled_q = output.q_values.float()[
        torch.arange(output.q_values.shape[0], device=self.device),
        policy_samples,
    ]
    expected_q = (output.q_values.float() * prob).sum(dim=-1)
    centered_q = sampled_q - expected_q
    entropy = -(prob * log_prob).sum(dim=-1)
    entropy_coef = self._entropy_coef()
    sampled_log_prob = log_prob[
        torch.arange(log_prob.shape[0], device=self.device),
        policy_samples,
    ]
    policy_loss = -(centered_q.detach() * sampled_log_prob) + entropy_coef * (-entropy)

    per_sample_loss = q_sac_loss + policy_loss
    loss = (loss_weights * per_sample_loss).mean()

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
    self.optimizer.step()
    self._maybe_update_target()
    self.gradient_steps += 1
    self.cycle_gradient_steps += 1

    priorities = torch.sqrt(dqn_loss.detach() + 1e-10).cpu().numpy()
    action_histogram = torch.bincount(
        policy_samples.detach(),
        minlength=self.config.num_actions,
    ).float()
    action_histogram = action_histogram / action_histogram.sum().clamp_min(1.0)
    return {
        "TotalLoss": float(loss.detach().cpu()),
        "DQNLoss": float(dqn_loss.mean().detach().cpu()),
        "TD Error": float(td_error.mean().detach().cpu()),
        "SPRLoss": float(spr_loss.mean().detach().cpu()),
        "PolicyLoss": float(policy_loss.mean().detach().cpu()),
        "Entropy": float(entropy.mean().detach().cpu()),
        "EntropyCoef": float(entropy_coef),
        "Alpha": float(self.online_network.entropy_scale().detach().cpu()),
        "PolicyLogitAbsMean": float(policy_logits.detach().abs().mean().cpu()),
        "PolicyLogitAbsMax": float(policy_logits.detach().abs().max().cpu()),
        "PolicyActionMaxProb": float(prob.detach().max(dim=-1).values.mean().cpu()),
        "PolicySampleActionHistogram": action_histogram.cpu().numpy(),
        "GradNorm": float(torch.as_tensor(grad_norm).detach().cpu()),
        "priorities": priorities,
    }

  def _entropy_coef(self) -> float:
    steps_left = self.config.entropy_decay_period - self.training_steps
    bonus = (
        (self.config.entropy_initial_coef - self.config.entropy_final_coef)
        * steps_left
        / self.config.entropy_decay_period
    )
    bonus = min(
        max(bonus, 0.0),
        self.config.entropy_initial_coef - self.config.entropy_final_coef,
    )
    return self.config.entropy_final_coef + bonus

  def _spr_loss(
      self,
      predictions: torch.Tensor,
      targets: torch.Tensor,
      same_trajectory: torch.Tensor,
  ) -> torch.Tensor:
    if self.config.spr_weight <= 0 or predictions.ndim != 3:
      return torch.zeros(targets.shape[0], device=self.device)
    horizon = min(predictions.shape[1], targets.shape[1], same_trajectory.shape[1])
    predictions = predictions[:, :horizon]
    targets = targets[:, :horizon]
    mask = same_trajectory[:, :horizon].float()
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], 2, -1)
    targets = targets.reshape(targets.shape[0], targets.shape[1], 2, -1)
    predictions = F.normalize(predictions, p=2, dim=-1)
    targets = F.normalize(targets, p=2, dim=-1)
    loss = (predictions - targets).pow(2).sum(dim=(-1, -2))
    return (loss * mask).mean(dim=-1) * 0.5

  def select_action(self, state: np.ndarray | torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    network = self.target_network if self.config.target_action_selection else self.online_network
    noisy_eval_mode = eval_mode and not self.config.eval_noise
    network.eval() if eval_mode else network.train()
    state_tensor = self._state_to_tensor(state)
    with torch.inference_mode():
      logits, samples = network.get_policy(
          state_tensor,
          eval_mode=noisy_eval_mode,
          data_augmentation=False,
      )
      if eval_mode:
        return logits.argmax(dim=-1)
      if self.training_steps < self.config.min_replay_history:
        return torch.randint(
            0,
            self.config.num_actions,
            (state_tensor.shape[0],),
            device=self.device,
            generator=self.generator,
        )
      return samples
