"""Encoder registry for Atari 100K network backbones."""

from src.algorithms.atari100k.encoders.base import EncoderName
from src.algorithms.atari100k.encoders.base import InitializerName
from src.algorithms.atari100k.encoders.base import ResNet18Variant
from src.algorithms.atari100k.encoders.registry import make_encoder

__all__ = ["EncoderName", "InitializerName", "ResNet18Variant", "make_encoder"]
