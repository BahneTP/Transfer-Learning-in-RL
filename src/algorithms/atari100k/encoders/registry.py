"""Factory for Atari 100K encoders."""

from __future__ import annotations

from torch import nn

from src.algorithms.atari100k.encoders.base import EncoderName
from src.algorithms.atari100k.encoders.base import InitializerName
from src.algorithms.atari100k.encoders.base import ResNet18Variant
from src.algorithms.atari100k.encoders.cnn import ImpalaCNN
from src.algorithms.atari100k.encoders.cnn import RainbowCNN
from src.algorithms.atari100k.transfer_learning import DINOv2ViTS14Encoder
from src.algorithms.atari100k.transfer_learning import ResNet18Encoder


def make_encoder(
    *,
    encoder_type: EncoderName,
    input_channels: int,
    width_scale: int,
    initializer: InitializerName,
    resnet18_weights: str | None = None,
    resnet18_variant: ResNet18Variant = "resnet_layer3_reduced",
    dinov2_weights: str | None = None,
    dinov2_output_block: int = 12,
    dinov2_output_mode: str = "single_block",
    dinov2_mix_blocks: tuple[int, ...] | list[int] | None = None,
) -> nn.Module:
  if encoder_type == "dqn":
    return RainbowCNN(width_scale=width_scale, initializer=initializer)
  if encoder_type == "impala":
    return ImpalaCNN(
        input_channels=input_channels,
        width_scale=width_scale,
        initializer=initializer,
    )
  if encoder_type == "resnet18":
    return ResNet18Encoder(
        input_channels=input_channels,
        weights=resnet18_weights,
        variant=resnet18_variant,
        initializer=initializer,
    )
  if encoder_type == "dinov2_vits14":
    return DINOv2ViTS14Encoder(
        input_channels=input_channels,
        weights=dinov2_weights,
        output_block=dinov2_output_block,
        output_mode=dinov2_output_mode,
        mix_blocks=dinov2_mix_blocks,
        initializer=initializer,
    )
  raise NotImplementedError(f"Unsupported encoder_type {encoder_type}")
