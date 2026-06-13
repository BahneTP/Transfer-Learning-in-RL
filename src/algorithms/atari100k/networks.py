"""Network blocks for Atari 100K Rainbow-style agents."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


EncoderName = Literal["dqn", "impala"]


@dataclasses.dataclass
class RainbowOutput:
    q_values: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor


def _apply_initializer(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def renormalize(tensor: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    flat = tensor.reshape(tensor.shape[0], -1)
    max_value = flat.max(dim=-1, keepdim=True).values
    min_value = flat.min(dim=-1, keepdim=True).values
    return ((flat - min_value) / (max_value - min_value + 1e-5)).reshape(shape)


def preprocess_observation(x: torch.Tensor) -> torch.Tensor:
    """Convert TorchRL/Atari observations to float CHW tensors in [0, 1]."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] in (1, 3, 4) and x.shape[-3] not in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)
    x = x.float()
    if x.max().detach() > 1.5:
        x = x / 255.0
    return x


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
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
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

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
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class FeatureLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, noisy: bool) -> None:
        super().__init__()
        self.noisy = noisy
        if noisy:
            self.net: nn.Module = NoisyLinear(in_features, out_features)
        else:
            self.net = nn.Linear(in_features, out_features)
            _apply_initializer(self.net)

    def forward(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
        if self.noisy:
            return self.net(x, eval_mode=eval_mode)
        return self.net(x)


class LinearHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_actions: int,
        num_atoms: int,
        *,
        noisy: bool,
        dueling: bool,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.advantage = FeatureLayer(in_features, num_actions * num_atoms, noisy=noisy)
        self.value = FeatureLayer(in_features, num_atoms, noisy=noisy) if dueling else None

    def forward(self, x: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
        adv = self.advantage(x, eval_mode=eval_mode).view(x.shape[0], self.num_actions, self.num_atoms)
        if self.dueling and self.value is not None:
            value = self.value(x, eval_mode=eval_mode).view(x.shape[0], 1, self.num_atoms)
            return value + (adv - adv.mean(dim=1, keepdim=True))
        return adv


class RainbowCNN(nn.Module):
    def __init__(self, input_channels: int = 4, width_scale: int = 1) -> None:
        super().__init__()
        dims = [int(dim * width_scale) for dim in (32, 64, 64)]
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.output_channels = dims[-1]
        self.apply(_apply_initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RainbowDQNNetwork(nn.Module):
    """Rainbow/DER convolutional C51 network."""

    def __init__(
        self,
        *,
        num_actions: int,
        num_atoms: int = 51,
        noisy: bool = True,
        dueling: bool = True,
        distributional: bool = True,
        encoder_type: EncoderName = "dqn",
        hidden_dim: int = 512,
        width_scale: int = 1,
        renormalize_output: bool = False,
        input_channels: int = 4,
    ) -> None:
        super().__init__()
        if encoder_type != "dqn":
            raise NotImplementedError("DER currently uses the DQN/Rainbow encoder.")
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.distributional = distributional
        self.renormalize_output = renormalize_output
        self.encoder = RainbowCNN(input_channels=input_channels, width_scale=width_scale)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            flat_dim = int(self.encoder(dummy).reshape(1, -1).shape[-1])
        self.projection = FeatureLayer(flat_dim, hidden_dim, noisy=noisy)
        self.head = LinearHead(hidden_dim, num_actions, num_atoms, noisy=noisy, dueling=dueling)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(preprocess_observation(x))
        if self.renormalize_output:
            latent = renormalize(latent)
        return latent

    def forward(self, x: torch.Tensor, support: torch.Tensor, *, eval_mode: bool = False) -> RainbowOutput:
        latent = self.encode(x).reshape(x.shape[0] if x.ndim == 4 else 1, -1)
        hidden = F.relu(self.projection(latent, eval_mode=eval_mode))
        logits = self.head(hidden, eval_mode=eval_mode)
        if self.distributional:
            probabilities = F.softmax(logits, dim=-1)
            q_values = (probabilities * support.view(1, 1, -1)).sum(dim=-1)
        else:
            probabilities = torch.ones_like(logits) / logits.shape[-1]
            q_values = logits.squeeze(-1)
        return RainbowOutput(q_values=q_values, logits=logits, probabilities=probabilities)
