"""Transfer-learning building blocks for Atari 100K agents."""

from __future__ import annotations

import math
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

from src.algorithms.atari100k.encoders.base import InitializerName
from src.algorithms.atari100k.encoders.base import ResNet18Variant
from src.algorithms.atari100k.encoders.base import apply_initializer


class AttentiveProbe(nn.Module):
  """Small trainable attention pooling head over spatial encoder features."""

  def __init__(
      self,
      *,
      in_channels: int,
      out_features: int,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    self.query = nn.Parameter(torch.empty(in_channels))
    self.value = nn.Linear(in_channels, out_features)
    self.score = nn.Linear(in_channels, 1)
    nn.init.normal_(self.query, std=1.0 / math.sqrt(in_channels))
    apply_initializer(self.value, initializer)
    apply_initializer(self.score, initializer)

  def forward(self, spatial_latent: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    del eval_mode
    tokens = spatial_latent.flatten(2).transpose(1, 2)
    scores = self.score(tokens + self.query.view(1, 1, -1)).squeeze(-1)
    weights = scores.softmax(dim=-1)
    pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
    return self.value(pooled)


class LoRALinear(nn.Module):
  """Low-rank adapter wrapper for a frozen linear layer."""

  def __init__(
      self,
      base: nn.Linear,
      *,
      rank: int,
      alpha: float,
      dropout: float,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    if rank <= 0:
      raise ValueError("LoRA rank must be positive.")
    self.base = base
    for parameter in self.base.parameters():
      parameter.requires_grad = False
    self.lora_down = nn.Linear(base.in_features, rank, bias=False)
    self.lora_up = nn.Linear(rank, base.out_features, bias=False)
    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    self.scaling = alpha / rank
    apply_initializer(self.lora_down, initializer)
    nn.init.zeros_(self.lora_up.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
  """Low-rank adapter wrapper for a frozen 2D convolution."""

  def __init__(
      self,
      base: nn.Conv2d,
      *,
      rank: int,
      alpha: float,
      dropout: float,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    if rank <= 0:
      raise ValueError("LoRA rank must be positive.")
    if base.groups != 1:
      raise ValueError("LoRAConv2d only supports groups=1 convolutions.")
    self.base = base
    for parameter in self.base.parameters():
      parameter.requires_grad = False
    self.lora_down = nn.Conv2d(
        base.in_channels,
        rank,
        kernel_size=base.kernel_size,
        stride=base.stride,
        padding=base.padding,
        dilation=base.dilation,
        bias=False,
    )
    self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
    self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    self.scaling = alpha / rank
    apply_initializer(self.lora_down, initializer)
    nn.init.zeros_(self.lora_up.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


def apply_lora_adapters(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    initializer: InitializerName = "xavier_uniform",
) -> int:
  """Recursively replace Linear/Conv2d children with frozen LoRA wrappers."""

  replacements = 0
  for name, child in list(module.named_children()):
    if isinstance(child, (LoRALinear, LoRAConv2d)):
      continue
    if isinstance(child, nn.Linear):
      setattr(
          module,
          name,
          LoRALinear(
              child,
              rank=rank,
              alpha=alpha,
              dropout=dropout,
              initializer=initializer,
          ),
      )
      replacements += 1
    elif isinstance(child, nn.Conv2d):
      setattr(
          module,
          name,
          LoRAConv2d(
              child,
              rank=rank,
              alpha=alpha,
              dropout=dropout,
              initializer=initializer,
          ),
      )
      replacements += 1
    else:
      replacements += apply_lora_adapters(
          child,
          rank=rank,
          alpha=alpha,
          dropout=dropout,
          initializer=initializer,
      )
  return replacements


class ResNet18Encoder(nn.Module):
  """ResNet-18 trunk adapted for Atari frame stacks.

  The encoder keeps the spatial feature map instead of ResNet's average-pool
  and classifier, matching the interface used by the DQN and IMPALA encoders.
  """

  def __init__(
      self,
      *,
      input_channels: int = 4,
      weights: str | None = None,
      variant: ResNet18Variant = "resnet_layer3_reduced",
      use_input_adapter: bool = False,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    resolved_weights = self._resolve_weights(weights)
    backbone = resnet18(weights=resolved_weights)
    normalization_channels = 3 if use_input_adapter else input_channels
    if resolved_weights is None:
      mean = torch.zeros(normalization_channels)
      std = torch.ones(normalization_channels)
    else:
      image_mean = torch.as_tensor(resolved_weights.transforms().mean)
      image_std = torch.as_tensor(resolved_weights.transforms().std)
      if use_input_adapter:
        mean = image_mean
        std = image_std
      else:
        mean = image_mean.mean().repeat(input_channels)
        std = image_std.mean().repeat(input_channels)
    self.input_adapter = (
        self._make_input_adapter(input_channels)
        if use_input_adapter and input_channels != backbone.conv1.in_channels
        else None
    )
    self.register_buffer("input_mean", mean.view(1, normalization_channels, 1, 1))
    self.register_buffer("input_std", std.view(1, normalization_channels, 1, 1))
    first_conv = backbone.conv1 if self.input_adapter is not None else self._adapt_first_conv(
        backbone.conv1,
        input_channels,
    )
    self.stem = nn.Sequential(
        first_conv,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
    )
    if variant == "resnet_full":
      self.layers = nn.Sequential(
          backbone.layer1,
          backbone.layer2,
          backbone.layer3,
          backbone.layer4,
      )
      self.reducer = None
      self.output_channels = 512
    elif variant == "resnet_layer3_flattened":
      self.layers = nn.Sequential(
          backbone.layer1,
          backbone.layer2,
          backbone.layer3,
      )
      self.reducer = None
      self.output_channels = 256
    elif variant == "resnet_layer3_reduced":
      self.layers = nn.Sequential(
          backbone.layer1,
          backbone.layer2,
          backbone.layer3,
      )
      self.reducer = nn.Conv2d(256, 64, kernel_size=1)
      self.output_channels = 64
    else:
      raise ValueError(f"Unsupported resnet18 variant {variant!r}.")
    self.variant = variant
    self.use_input_adapter = use_input_adapter
    if self.reducer is not None:
      apply_initializer(self.reducer, initializer)

  def _make_input_adapter(self, input_channels: int) -> nn.Conv2d:
    adapter = nn.Conv2d(input_channels, 3, kernel_size=1)
    with torch.no_grad():
      adapter.weight.fill_(1.0 / input_channels)
      adapter.bias.zero_()
    return adapter

  def _resolve_weights(self, weights: str | None) -> ResNet18_Weights | None:
    if weights is None or str(weights).lower() in {"", "none", "false"}:
      return None
    if str(weights).lower() in {"default", "imagenet", "imagenet1k"}:
      return ResNet18_Weights.DEFAULT
    return ResNet18_Weights[weights]

  def _adapt_first_conv(self, conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    adapted = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
      if input_channels == conv.in_channels:
        adapted.weight.copy_(conv.weight)
      else:
        gray_weight = conv.weight.mean(dim=1, keepdim=True)
        adapted.weight.copy_(gray_weight.repeat(1, input_channels, 1, 1))
        adapted.weight.mul_(conv.in_channels / input_channels)
      if conv.bias is not None and adapted.bias is not None:
        adapted.bias.copy_(conv.bias)
    return adapted

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.input_adapter is not None:
      x = self.input_adapter(x)
    x = (x - self.input_mean) / self.input_std
    x = self.layers(self.stem(x))
    if self.reducer is not None:
      x = self.reducer(x)
    return x
