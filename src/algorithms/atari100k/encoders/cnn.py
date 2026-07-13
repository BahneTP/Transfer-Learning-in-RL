"""Convolutional Atari encoders."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.algorithms.atari100k.encoders.base import InitializerName
from src.algorithms.atari100k.encoders.base import apply_initializer


class RainbowCNN(nn.Module):
  def __init__(
      self,
      *,
      width_scale: int = 1,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    dims = [int(dim * width_scale) for dim in (32, 64, 64)]
    self.layers = nn.Sequential(
        nn.Conv2d(4, dims[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(dims[0], dims[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1),
        nn.ReLU(),
    )
    self.output_channels = dims[-1]
    self.apply(lambda module: apply_initializer(module, initializer))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers(x)


class ResidualStage(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      *,
      num_blocks: int = 2,
      use_max_pooling: bool = True,
      dropout: float = 0.0,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.use_max_pooling = use_max_pooling
    self.blocks = nn.ModuleList()
    for _ in range(num_blocks):
      self.blocks.append(nn.ModuleList([
          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      ]))
    self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    self.apply(lambda module: apply_initializer(module, initializer))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.proj(x)
    if self.use_max_pooling:
      out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
    for conv1, conv2 in self.blocks:
      residual = out
      out = F.relu(out)
      out = self.dropout(out)
      out = conv1(out)
      out = F.relu(out)
      out = conv2(out)
      out = out + residual
    return out


class ImpalaCNN(nn.Module):
  def __init__(
      self,
      *,
      input_channels: int = 4,
      width_scale: int = 1,
      dims: tuple[int, ...] = (16, 32, 32),
      num_blocks: int = 2,
      dropout: float = 0.0,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    stages = []
    in_channels = input_channels
    for width in dims:
      out_channels = int(width * width_scale)
      stages.append(
          ResidualStage(
              in_channels,
              out_channels,
              num_blocks=num_blocks,
              dropout=dropout,
              initializer=initializer,
          )
      )
      in_channels = out_channels
    self.stages = nn.Sequential(*stages)
    self.output_channels = in_channels

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.relu(self.stages(x))
