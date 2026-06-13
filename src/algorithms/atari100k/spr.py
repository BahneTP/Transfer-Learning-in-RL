"""Self-Predictive Representations (SPR) for Atari 100K."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.algorithms.atari100k.der import DERAtari100KAlgorithm
from src.algorithms.atari100k.rl import categorical_target


class SPRAtari100KAlgorithm(DERAtari100KAlgorithm):
    """SPR extends DER with a latent transition-model prediction loss."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        gamma: float = 0.99,
        update_horizon: int = 10,
        min_replay_history: int = 2000,
        target_update_period: int = 1,
        epsilon_train: float = 0.0,
        epsilon_eval: float = 0.001,
        epsilon_decay_period: int = 2001,
        replay_ratio: int = 64,
        batch_size: int = 32,
        spr_weight: float = 5.0,
        jumps: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            device,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            target_update_period=target_update_period,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            noisy=True,
            dueling=True,
            double_dqn=True,
            **kwargs,
        )
        self.spr_weight = spr_weight
        self.jumps = jumps

    def _train_one_update(self) -> dict[str, torch.Tensor]:
        sample = self.replay.sample_sequence(self.batch_size, self.device, jumps=self.jumps)
        assert sample.future_states is not None
        assert sample.rollout_actions is not None
        assert sample.same_trajectory is not None
        with torch.no_grad():
            next_online = self.online_network(sample.next_states, self.support, eval_mode=True)
            next_target = self.target_network(sample.next_states, self.support, eval_mode=True)
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
            flat_future = sample.future_states.reshape(-1, *sample.future_states.shape[2:])
            spr_targets = self.target_network.encode_project(flat_future, eval_mode=True)
            spr_targets = spr_targets.reshape(sample.future_states.shape[0], sample.future_states.shape[1], -1)

        output = self.online_network(
            sample.states,
            self.support,
            actions=sample.rollout_actions,
            do_rollout=True,
            eval_mode=False,
        )
        chosen_logits = output.logits[torch.arange(sample.actions.shape[0], device=self.device), sample.actions]
        dqn_loss = -(target * F.log_softmax(chosen_logits, dim=-1)).sum(dim=-1)
        spr_loss = self._spr_loss(output.latent, spr_targets, sample.same_trajectory)
        per_sample_loss = dqn_loss + self.spr_weight * spr_loss
        loss = (sample.weights * per_sample_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._maybe_update_target()
        priorities = torch.sqrt(dqn_loss.detach() + 1e-10).cpu().numpy()
        self.replay.set_priority(sample.indices, priorities)
        self.gradient_steps += 1
        return {
            "loss": loss.detach(),
            "grad_norm": torch.as_tensor(grad_norm, device=self.device),
            "spr_loss": spr_loss.mean().detach(),
        }

    def step(self, batch):
        metrics = super().step(batch)
        return metrics

    def _spr_loss(
        self,
        predictions: torch.Tensor | None,
        targets: torch.Tensor,
        same_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        if predictions is None or predictions.numel() == 0:
            return torch.zeros(targets.shape[0], device=self.device)
        horizon = min(predictions.shape[1], targets.shape[1], same_trajectory.shape[1])
        predictions = F.normalize(predictions[:, :horizon], p=2, dim=-1)
        targets = F.normalize(targets[:, :horizon], p=2, dim=-1)
        mask = same_trajectory[:, :horizon].float()
        loss = (predictions - targets).pow(2).sum(dim=-1)
        return (loss * mask).mean(dim=-1) * 0.5
