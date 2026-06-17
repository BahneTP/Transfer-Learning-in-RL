"""Self-Predictive Representations (SPR) agent preset in PyTorch."""

from __future__ import annotations

import dataclasses

from src.algorithms.atari100k.bbf import BBFAgent
from src.algorithms.atari100k.bbf import BBFConfig


@dataclasses.dataclass
class SPRConfig(BBFConfig):
  """Reference-style SPR configuration.

  The implementation reuses BBF's SPR training path, but the preset disables
  BBF's periodic reset machinery and uses the DQN-scale encoder/head from SPR.
  """


class SPRAgent(BBFAgent):
  """SPR is BBF's auxiliary prediction path without BBF-specific resets."""

  config: SPRConfig


@dataclasses.dataclass
class SRSPRConfig(BBFConfig):
  """Shrink-and-Reset SPR configuration."""


class SRSPRAgent(BBFAgent):
  """SR-SPR adds BBF-style shrink-and-reset to the SPR backbone."""

  config: SRSPRConfig
