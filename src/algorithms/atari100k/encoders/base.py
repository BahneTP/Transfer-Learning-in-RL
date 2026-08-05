"""Shared encoder types and initialization helpers."""

from __future__ import annotations

from typing import Literal

from torch import nn


InitializerName = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
]
EncoderName = Literal["dqn", "impala", "resnet18", "dinov2_vits14"]
ResNet18Variant = Literal[
    "resnet_full",
    "resnet_layer1_reduced",
    "resnet_layer2_reduced",
    "resnet_layer3_flattened",
    "resnet_layer3_reduced",
    "resnet_layer4_reduced",
]


def apply_initializer(module: nn.Module, initializer: InitializerName) -> None:
  if isinstance(module, (nn.Conv2d, nn.Linear)):
    if initializer == "xavier_uniform":
      nn.init.xavier_uniform_(module.weight)
    elif initializer == "xavier_normal":
      nn.init.xavier_normal_(module.weight)
    elif initializer == "kaiming_uniform":
      nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    elif initializer == "kaiming_normal":
      nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif initializer == "orthogonal":
      nn.init.orthogonal_(module.weight)
    else:
      raise NotImplementedError(f"Unsupported initializer: {initializer}")
    if module.bias is not None:
      nn.init.zeros_(module.bias)
