"""Shared PyTorch network building blocks for DER and BBF."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from src.algorithms.atari100k.encoders import EncoderName
from src.algorithms.atari100k.encoders import InitializerName
from src.algorithms.atari100k.encoders import ResNet18Variant
from src.algorithms.atari100k.encoders import make_encoder
from src.algorithms.atari100k.transfer_learning import AttentiveProbe
from src.algorithms.atari100k.encoders.base import apply_initializer


ProbeName = Literal["flatten", "attentive"]


@dataclasses.dataclass
class SPRNetworkOutput:
  q_values: torch.Tensor
  logits: torch.Tensor | None
  probabilities: torch.Tensor | None
  latent: torch.Tensor
  representation: torch.Tensor


def renormalize(tensor: torch.Tensor) -> torch.Tensor:
  shape = tensor.shape
  flat = tensor.reshape(tensor.shape[0], -1)
  max_value = flat.max(dim=-1, keepdim=True).values
  min_value = flat.min(dim=-1, keepdim=True).values
  return ((flat - min_value) / (max_value - min_value + 1e-5)).reshape(shape)


def process_inputs(
    x: torch.Tensor,
    *,
    data_augmentation: bool = False,
    pad: int = 4,
) -> torch.Tensor:
  out = x.float() / 255.0
  if data_augmentation:
    out = F.pad(out, (pad, pad, pad, pad), mode="replicate")
    crop_h = x.shape[-2]
    crop_w = x.shape[-1]
    max_y = out.shape[-2] - crop_h
    max_x = out.shape[-1] - crop_w
    ys = torch.randint(0, max_y + 1, (out.shape[0],), device=out.device)
    xs = torch.randint(0, max_x + 1, (out.shape[0],), device=out.device)
    batch_indices = torch.arange(out.shape[0], device=out.device)[:, None, None]
    y_offsets = ys[:, None, None] + torch.arange(crop_h, device=out.device)[None, :, None]
    x_offsets = xs[:, None, None] + torch.arange(crop_w, device=out.device)[None, None, :]
    out = out.permute(0, 2, 3, 1)[batch_indices, y_offsets, x_offsets].permute(0, 3, 1, 2)
    noise = 1.0 + 0.05 * torch.randn((out.shape[0], 1, 1, 1), device=out.device).clamp(-2.0, 2.0)
    out = out * noise
  return out


class NoisyLinear(nn.Module):
  """Factorized Gaussian noisy linear layer."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      std_init: float = 0.5,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
    self.register_buffer("bias_epsilon", torch.zeros(out_features))
    self.std_init = std_init
    self.initializer = initializer
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self) -> None:
    bound = 1 / math.sqrt(self.in_features)
    if self.initializer == "xavier_uniform":
      nn.init.xavier_uniform_(self.weight_mu)
    elif self.initializer == "xavier_normal":
      nn.init.xavier_normal_(self.weight_mu)
    elif self.initializer == "kaiming_uniform":
      nn.init.kaiming_uniform_(self.weight_mu, nonlinearity="relu")
    elif self.initializer == "kaiming_normal":
      nn.init.kaiming_normal_(self.weight_mu, nonlinearity="relu")
    elif self.initializer == "orthogonal":
      nn.init.orthogonal_(self.weight_mu)
    else:
      raise NotImplementedError(f"Unsupported initializer: {self.initializer}")
    nn.init.uniform_(self.bias_mu, -bound, bound)
    nn.init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
    nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.in_features))

  def _scale_noise(self, size: int) -> torch.Tensor:
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign() * x.abs().sqrt()

  def reset_noise(self) -> None:
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    if eval_mode:
      weight = self.weight_mu
      bias = self.bias_mu
    else:
      epsilon_in = self._scale_noise(self.in_features)
      epsilon_out = self._scale_noise(self.out_features)
      weight_epsilon = epsilon_out.outer(epsilon_in)
      bias_epsilon = epsilon_out
      weight = self.weight_mu + self.weight_sigma * weight_epsilon
      bias = self.bias_mu + self.bias_sigma * bias_epsilon
    return F.linear(x, weight, bias)


class FeatureLayer(nn.Module):
  def __init__(
      self,
      *,
      noisy: bool,
      in_features: int,
      out_features: int,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    self.noisy = noisy
    self.net: nn.Module
    if noisy:
      self.net = NoisyLinear(
          in_features,
          out_features,
          initializer=initializer,
      )
    else:
      self.net = nn.Linear(in_features, out_features)
      apply_initializer(self.net, initializer)

  def forward(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    if self.noisy:
      return self.net(x, eval_mode=eval_mode)
    return self.net(x)


class LinearHead(nn.Module):
  def __init__(
      self,
      *,
      noisy: bool,
      dueling: bool,
      in_features: int,
      num_actions: int,
      num_atoms: int,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    self.dueling = dueling
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.advantage = FeatureLayer(
        noisy=noisy,
        in_features=in_features,
        out_features=num_actions * num_atoms,
        initializer=initializer,
    )
    self.value = None
    if dueling:
      self.value = FeatureLayer(
          noisy=noisy,
          in_features=in_features,
          out_features=num_atoms,
          initializer=initializer,
      )

  def forward(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    adv = self.advantage(x, eval_mode=eval_mode).view(x.shape[0], self.num_actions, self.num_atoms)
    if self.dueling and self.value is not None:
      value = self.value(x, eval_mode=eval_mode).view(x.shape[0], 1, self.num_atoms)
      return value + (adv - adv.mean(dim=1, keepdim=True))
    return adv


class ConvTransitionCell(nn.Module):
  def __init__(
      self,
      *,
      num_actions: int,
      latent_dim: int,
      renormalize_output: bool,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    self.num_actions = num_actions
    self.renormalize_output = renormalize_output
    self.conv1 = nn.Conv2d(latent_dim + num_actions, latent_dim, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)
    self.apply(lambda module: apply_initializer(module, initializer))

  def forward(self, x: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = x.shape
    action_onehot = F.one_hot(action.long(), num_classes=self.num_actions).float()
    action_plane = action_onehot[:, :, None, None].expand(batch, self.num_actions, height, width)
    out = torch.cat([x, action_plane], dim=1)
    out = F.relu(self.conv1(out))
    out = F.relu(self.conv2(out))
    if self.renormalize_output:
      out = renormalize(out)
    return out, out


class TransitionModel(nn.Module):
  def __init__(
      self,
      *,
      num_actions: int,
      latent_dim: int,
      renormalize_output: bool,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    self.cell = ConvTransitionCell(
        num_actions=num_actions,
        latent_dim=latent_dim,
        renormalize_output=renormalize_output,
        initializer=initializer,
    )

  def forward(self, x: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latents = []
    current = x
    for t in range(actions.shape[1]):
      current, pred = self.cell(current, actions[:, t])
      latents.append(pred)
    return current, torch.stack(latents, dim=1)


class RainbowDQNNetwork(nn.Module):
  """Common PyTorch backbone for DER and BBF family agents."""

  def __init__(
      self,
      *,
      num_actions: int,
      num_atoms: int,
      noisy: bool,
      dueling: bool,
      distributional: bool,
      renormalize_output: bool = False,
      encoder_type: EncoderName = "dqn",
      hidden_dim: int = 512,
      width_scale: int = 1,
      initializer: InitializerName = "xavier_uniform",
      input_channels: int = 4,
      resnet18_weights: str | None = None,
      resnet18_variant: ResNet18Variant = "resnet_layer3_reduced",
      dinov2_weights: str | None = None,
      dinov2_output_block: int = 12,
      dinov2_output_mode: str = "single_block",
      dinov2_mix_blocks: tuple[int, ...] | list[int] | None = None,
      probe_type: ProbeName = "flatten",
  ) -> None:
    super().__init__()
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.distributional = distributional
    self.renormalize_output = renormalize_output
    self.encoder = make_encoder(
        encoder_type=encoder_type,
        input_channels=input_channels,
        width_scale=width_scale,
        initializer=initializer,
        resnet18_weights=resnet18_weights,
        resnet18_variant=resnet18_variant,
        dinov2_weights=dinov2_weights,
        dinov2_output_block=dinov2_output_block,
        dinov2_output_mode=dinov2_output_mode,
        dinov2_mix_blocks=dinov2_mix_blocks,
    )
    latent_dim = self.encoder.output_channels
    self.transition_model = TransitionModel(
        num_actions=num_actions,
        latent_dim=latent_dim,
        renormalize_output=renormalize_output,
        initializer=initializer,
    )
    self.projection = None
    self.projection_out_dim: int | None = None
    self.latent_dim = latent_dim
    self.hidden_dim = hidden_dim
    self.noisy = noisy
    self.dueling = dueling
    self.initializer = initializer
    self.input_channels = input_channels
    self.probe_type = probe_type
    self.head: LinearHead | None = None
    self.predictor: nn.Linear | None = None
    self.jepa_action_embedding: nn.Embedding | None = None
    self.jepa_predictor: nn.Sequential | None = None

  def _ensure_head(self, representation_dim: int, device: torch.device) -> None:
    if self.projection is None:
      if self.probe_type == "flatten":
        self.projection = FeatureLayer(
            noisy=self.noisy,
            in_features=representation_dim,
            out_features=self.hidden_dim,
            initializer=self.initializer,
        )
      elif self.probe_type == "attentive":
        self.projection = AttentiveProbe(
            in_channels=self.latent_dim,
            out_features=self.hidden_dim,
            initializer=self.initializer,
        )
      else:
        raise NotImplementedError(f"Unsupported probe_type {self.probe_type}")
      self.projection_out_dim = self.hidden_dim
      self.predictor = nn.Linear(self.hidden_dim, self.hidden_dim)
      apply_initializer(self.predictor, self.initializer)
      self.head = LinearHead(
          noisy=self.noisy,
          dueling=self.dueling,
          in_features=self.hidden_dim,
          num_actions=self.num_actions,
          num_atoms=self.num_atoms,
          initializer=self.initializer,
      )
      self.add_module("projection_layer", self.projection)
      self.add_module("predictor_layer", self.predictor)
      self.add_module("head_layer", self.head)
      self.projection.to(device)
      self.predictor.to(device)
      self.head.to(device)

  def ensure_jepa_predictor(self, *, action_dim: int = 64, device: torch.device) -> None:
    if self.jepa_predictor is not None:
      return
    self.jepa_action_embedding = nn.Embedding(self.num_actions, action_dim)
    self.jepa_predictor = nn.Sequential(
        nn.Linear(self.hidden_dim + action_dim, self.hidden_dim),
        nn.GELU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
    )
    nn.init.normal_(self.jepa_action_embedding.weight, std=0.02)
    apply_initializer(self.jepa_predictor[0], self.initializer)
    apply_initializer(self.jepa_predictor[2], self.initializer)
    self.add_module("jepa_action_embedding_layer", self.jepa_action_embedding)
    self.add_module("jepa_predictor_layer", self.jepa_predictor)
    self.jepa_action_embedding.to(device)
    self.jepa_predictor.to(device)

  def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4 and x.shape[1] != self.input_channels and x.shape[-1] == self.input_channels:
      return x.permute(0, 3, 1, 2).contiguous()
    return x

  def encode(
      self,
      x: torch.Tensor,
      *,
      eval_mode: bool = False,
      data_augmentation: bool = False,
  ) -> torch.Tensor:
    del eval_mode
    processed = self.preprocess(x, data_augmentation=data_augmentation)
    return self.encode_processed(processed)

  def preprocess(
      self,
      x: torch.Tensor,
      *,
      data_augmentation: bool = False,
  ) -> torch.Tensor:
    return process_inputs(self._to_nchw(x), data_augmentation=data_augmentation)

  def encode_processed(self, processed: torch.Tensor) -> torch.Tensor:
    latent = self.encoder(processed)
    if self.renormalize_output:
      latent = renormalize(latent)
    return latent

  def flatten_spatial_latent(self, spatial_latent: torch.Tensor) -> torch.Tensor:
    return spatial_latent.reshape(spatial_latent.shape[0], -1)

  def project(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    self._ensure_head(x.shape[-1], x.device)
    assert self.projection is not None
    if self.probe_type != "flatten":
      raise ValueError("project() only accepts flat features for probe_type='flatten'.")
    return self.projection(x, eval_mode=eval_mode)

  def project_latent(self, latent: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    representation_dim = self.flatten_spatial_latent(latent).shape[-1]
    self._ensure_head(representation_dim, latent.device)
    assert self.projection is not None
    if self.probe_type == "attentive":
      return self.projection(latent, eval_mode=eval_mode)
    return self.projection(
        self.flatten_spatial_latent(latent),
        eval_mode=eval_mode,
    )

  def encode_project(
      self,
      x: torch.Tensor,
      *,
      eval_mode: bool = False,
      data_augmentation: bool = False,
  ) -> torch.Tensor:
    latent = self.encode(x, eval_mode=eval_mode, data_augmentation=data_augmentation)
    return self.encode_project_from_latent(latent, eval_mode=eval_mode)

  def encode_jepa_latent(
      self,
      x: torch.Tensor,
      *,
      eval_mode: bool = False,
      data_augmentation: bool = False,
  ) -> torch.Tensor:
    return F.relu(
        self.encode_project(
            x,
            eval_mode=eval_mode,
            data_augmentation=data_augmentation,
        )
    )

  def predict_next_jepa_latent(
      self,
      latent: torch.Tensor,
      actions: torch.Tensor,
      *,
      action_dim: int = 64,
  ) -> torch.Tensor:
    self.ensure_jepa_predictor(action_dim=action_dim, device=latent.device)
    assert self.jepa_action_embedding is not None
    assert self.jepa_predictor is not None
    action_embedding = self.jepa_action_embedding(actions.reshape(-1).long())
    return self.jepa_predictor(torch.cat([latent, action_embedding], dim=-1))

  def encode_project_from_latent(
      self,
      latent: torch.Tensor,
      *,
      eval_mode: bool = False,
  ) -> torch.Tensor:
    return self.project_latent(latent, eval_mode=eval_mode)

  def spr_predict(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    projected = self.project(x, eval_mode=eval_mode)
    assert self.predictor is not None
    return self.predictor(projected)

  def spr_predict_from_latent(self, latent: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    projected = self.project_latent(latent, eval_mode=eval_mode)
    assert self.predictor is not None
    return self.predictor(projected)

  def spr_rollout(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    _, pred_latents = self.transition_model(latent, actions)
    batch, time, channels, height, width = pred_latents.shape
    flat = pred_latents.reshape(batch * time, channels, height, width)
    preds = self.spr_predict_from_latent(flat, eval_mode=True)
    return preds.reshape(batch, time, -1)

  def forward(
      self,
      x: torch.Tensor,
      support: torch.Tensor,
      *,
      actions: torch.Tensor | None = None,
      do_rollout: bool = False,
      eval_mode: bool = False,
      data_augmentation: bool = False,
  ) -> SPRNetworkOutput:
    latent = self.encode(x, eval_mode=eval_mode, data_augmentation=data_augmentation)
    return self.forward_from_latent(
        latent,
        support,
        actions=actions,
        do_rollout=do_rollout,
        eval_mode=eval_mode,
    )

  def forward_from_latent(
      self,
      latent: torch.Tensor,
      support: torch.Tensor,
      *,
      actions: torch.Tensor | None = None,
      do_rollout: bool = False,
      eval_mode: bool = False,
  ) -> SPRNetworkOutput:
    representation = self.flatten_spatial_latent(latent)
    projected = self.project_latent(latent, eval_mode=eval_mode)
    projected = F.relu(projected)
    assert self.head is not None
    logits = self.head(projected, eval_mode=eval_mode)
    rollout_latent: torch.Tensor
    if do_rollout and actions is not None:
      rollout_latent = self.spr_rollout(latent, actions)
    else:
      rollout_latent = latent
    if self.distributional:
      probabilities = logits.softmax(dim=-1)
      q_values = torch.sum(support.view(1, 1, -1) * probabilities, dim=-1)
      return SPRNetworkOutput(
          q_values=q_values,
          logits=logits,
          probabilities=probabilities,
          latent=rollout_latent,
          representation=representation,
      )
    q_values = logits.squeeze(-1)
    return SPRNetworkOutput(
        q_values=q_values,
        logits=None,
        probabilities=None,
        latent=rollout_latent,
        representation=representation,
    )


class SACRainbowDQNNetwork(RainbowDQNNetwork):
  """Rainbow/BBF backbone with the discrete SAC policy head from SAC-BBF."""

  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    self.policy_projection: FeatureLayer | None = None
    self.predict_policy: nn.Linear | None = None
    self.policy: nn.Linear | None = None
    self._log_alpha = nn.Parameter(torch.zeros(()))

  def _ensure_head(self, representation_dim: int, device: torch.device) -> None:
    super()._ensure_head(representation_dim, device)
    if self.policy_projection is None:
      self.policy_projection = FeatureLayer(
          noisy=self.noisy,
          in_features=representation_dim,
          out_features=self.hidden_dim,
          initializer=self.initializer,
      )
      self.predict_policy = nn.Linear(self.hidden_dim, self.hidden_dim)
      self.policy = nn.Linear(self.hidden_dim, self.num_actions)
      apply_initializer(self.predict_policy, self.initializer)
      apply_initializer(self.policy, self.initializer)
      self.add_module("policy_projection_layer", self.policy_projection)
      self.add_module("predict_policy_layer", self.predict_policy)
      self.add_module("policy_layer", self.policy)
      self.policy_projection.to(device)
      self.predict_policy.to(device)
      self.policy.to(device)

  def entropy_scale(self) -> torch.Tensor:
    return self._log_alpha.exp()

  def policy_logits_from_representation(
      self,
      representation: torch.Tensor,
      *,
      eval_mode: bool = False,
  ) -> torch.Tensor:
    self._ensure_head(representation.shape[-1], representation.device)
    assert self.policy_projection is not None
    assert self.policy is not None
    projected = self.policy_projection(representation, eval_mode=eval_mode)
    return self.policy(F.relu(projected))

  def policy_logits_from_latent(
      self,
      latent: torch.Tensor,
      *,
      eval_mode: bool = False,
  ) -> torch.Tensor:
    representation = self.flatten_spatial_latent(latent)
    return self.policy_logits_from_representation(representation, eval_mode=eval_mode)

  def get_policy(
      self,
      x: torch.Tensor,
      *,
      eval_mode: bool = False,
      data_augmentation: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    latent = self.encode(x, eval_mode=eval_mode, data_augmentation=data_augmentation)
    logits = self.policy_logits_from_latent(latent, eval_mode=eval_mode)
    samples = torch.distributions.Categorical(logits=logits).sample()
    return logits, samples

  def encode_project_from_latent(
      self,
      latent: torch.Tensor,
      *,
      eval_mode: bool = False,
  ) -> torch.Tensor:
    representation = self.flatten_spatial_latent(latent)
    self._ensure_head(representation.shape[-1], representation.device)
    assert self.policy_projection is not None
    return torch.cat(
        [
            self.project_latent(latent, eval_mode=eval_mode),
            self.policy_projection(representation, eval_mode=eval_mode),
        ],
        dim=-1,
    )

  def spr_predict(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    self._ensure_head(x.shape[-1], x.device)
    assert self.predictor is not None
    assert self.policy_projection is not None
    assert self.predict_policy is not None
    return torch.cat(
        [
            self.predictor(self.project(x, eval_mode=eval_mode)),
            self.predict_policy(self.policy_projection(x, eval_mode=eval_mode)),
        ],
        dim=-1,
    )

  def spr_predict_from_latent(self, latent: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    representation = self.flatten_spatial_latent(latent)
    self._ensure_head(representation.shape[-1], representation.device)
    assert self.predictor is not None
    assert self.policy_projection is not None
    assert self.predict_policy is not None
    return torch.cat(
        [
            self.predictor(self.project_latent(latent, eval_mode=eval_mode)),
            self.predict_policy(self.policy_projection(representation, eval_mode=eval_mode)),
        ],
        dim=-1,
    )
