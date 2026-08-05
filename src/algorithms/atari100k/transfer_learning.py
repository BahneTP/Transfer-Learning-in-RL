"""Transfer-learning building blocks for Atari 100K agents."""

from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

from src.algorithms.atari100k.encoders.base import InitializerName
from src.algorithms.atari100k.encoders.base import ResNet18Variant
from src.algorithms.atari100k.encoders.base import apply_initializer


class AttentiveProbe(nn.Module):
  """Lightweight self-attention probe over spatial encoder features."""

  def __init__(
      self,
      *,
      in_channels: int,
      out_features: int,
      initializer: InitializerName = "xavier_uniform",
      num_tokens: int = 36,
      num_heads: int = 4,
  ) -> None:
    super().__init__()
    if in_channels % num_heads != 0:
      raise ValueError("AttentiveProbe in_channels must be divisible by num_heads.")
    self.in_channels = in_channels
    self.num_tokens = num_tokens
    self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, in_channels))
    self.attention_norm = nn.LayerNorm(in_channels)
    self.attention = nn.MultiheadAttention(
        embed_dim=in_channels,
        num_heads=num_heads,
        batch_first=True,
    )
    self.mlp_norm = nn.LayerNorm(in_channels)
    self.mlp = nn.Sequential(
        nn.Linear(in_channels, in_channels * 2),
        nn.GELU(),
        nn.Linear(in_channels * 2, in_channels),
    )
    self.value = nn.Linear(num_tokens * in_channels, out_features)
    nn.init.normal_(self.position_embedding, std=0.02)
    apply_initializer(self.mlp[0], initializer)
    apply_initializer(self.mlp[2], initializer)
    apply_initializer(self.value, initializer)

  def forward(self, spatial_latent: torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    del eval_mode
    tokens = spatial_latent.flatten(2).transpose(1, 2)
    if tokens.shape[1] != self.num_tokens or tokens.shape[2] != self.in_channels:
      raise ValueError(
          "AttentiveProbe expects spatial features with "
          f"{self.in_channels} channels and {self.num_tokens} tokens, got "
          f"{tokens.shape[2]} channels and {tokens.shape[1]} tokens."
      )
    tokens = tokens + self.position_embedding
    attended, _ = self.attention(
        self.attention_norm(tokens),
        self.attention_norm(tokens),
        self.attention_norm(tokens),
        need_weights=False,
    )
    tokens = tokens + attended
    tokens = tokens + self.mlp(self.mlp_norm(tokens))
    return self.value(tokens.reshape(tokens.shape[0], -1))


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


class DINOv2PatchEmbed(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.proj(x)
    return x.flatten(2).transpose(1, 2)


class DINOv2LayerScale(nn.Module):
  def __init__(self, dim: int = 384) -> None:
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.gamma


class DINOv2Attention(nn.Module):
  def __init__(self, dim: int = 384, num_heads: int = 6) -> None:
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.qkv = nn.Linear(dim, dim * 3)
    self.proj = nn.Linear(dim, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, tokens, dim = x.shape
    qkv = self.qkv(x).reshape(
        batch, tokens, 3, self.num_heads, self.head_dim
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    attention = (query @ key.transpose(-2, -1)) * self.scale
    attention = attention.softmax(dim=-1)
    x = (attention @ value).transpose(1, 2).reshape(batch, tokens, dim)
    return self.proj(x)


class DINOv2MLP(nn.Module):
  def __init__(self, dim: int = 384, hidden_dim: int = 1536) -> None:
    super().__init__()
    self.fc1 = nn.Linear(dim, hidden_dim)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(hidden_dim, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc2(self.act(self.fc1(x)))


class DINOv2Block(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(384)
    self.attn = DINOv2Attention()
    self.ls1 = DINOv2LayerScale()
    self.norm2 = nn.LayerNorm(384)
    self.mlp = DINOv2MLP()
    self.ls2 = DINOv2LayerScale()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.ls1(self.attn(self.norm1(x)))
    x = x + self.ls2(self.mlp(self.norm2(x)))
    return x


class DINOv2ViTS14Encoder(nn.Module):
  """DINOv2 ViT-S/14 adapted to Atari frame stacks via a 4->3 input adapter."""

  def __init__(
      self,
      *,
      input_channels: int = 4,
      weights: str | None = "models/dinov2_vits14_pretrain.pth",
      output_block: int = 12,
      output_mode: str = "single_block",
      mix_blocks: tuple[int, ...] | list[int] | None = None,
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    if output_block < 1 or output_block > 12:
      raise ValueError("DINOv2 output_block must be in [1, 12].")
    if output_mode not in {"single_block", "layer_mix"}:
      raise ValueError("DINOv2 output_mode must be 'single_block' or 'layer_mix'.")
    self.output_mode = output_mode
    self.output_block = output_block
    self.mix_blocks = tuple(mix_blocks) if mix_blocks is not None else tuple(range(1, 13))
    if not self.mix_blocks:
      raise ValueError("DINOv2 mix_blocks must not be empty.")
    invalid_blocks = [block for block in self.mix_blocks if block < 1 or block > 12]
    if invalid_blocks:
      raise ValueError(f"DINOv2 mix_blocks must be in [1, 12], got {invalid_blocks}.")
    self.mix_logits = (
        nn.Parameter(torch.zeros(len(self.mix_blocks)))
        if self.output_mode == "layer_mix"
        else None
    )
    self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)
    self._init_input_adapter(input_channels)
    self.register_buffer(
        "input_mean",
        torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
    )
    self.register_buffer(
        "input_std",
        torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
    )
    self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
    self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 384))
    self.mask_token = nn.Parameter(torch.zeros(1, 384))
    self.patch_embed = DINOv2PatchEmbed()
    self.blocks = nn.ModuleList([DINOv2Block() for _ in range(12)])
    self.norm = nn.LayerNorm(384)
    self.reducer = nn.Conv2d(384, 64, kernel_size=1)
    self.output_channels = 64
    apply_initializer(self.reducer, initializer)
    self._load_weights(weights)

  def _init_input_adapter(self, input_channels: int) -> None:
    with torch.no_grad():
      self.input_adapter.weight.fill_(1.0 / input_channels)
      if self.input_adapter.bias is not None:
        self.input_adapter.bias.zero_()

  def _load_weights(self, weights: str | None) -> None:
    if weights is None or str(weights).lower() in {"", "none", "false"}:
      return
    path = Path(weights)
    state = torch.load(path, map_location="cpu")
    incompatible = self.load_state_dict(state, strict=False)
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    allowed_missing = {
        "input_adapter.weight",
        "input_adapter.bias",
        "input_mean",
        "input_std",
        "reducer.weight",
        "reducer.bias",
        "mix_logits",
    }
    if unexpected or missing - allowed_missing:
      raise RuntimeError(
          "Unexpected DINOv2 checkpoint mismatch: "
          f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
      )

  def _position_embedding(self, height: int, width: int) -> torch.Tensor:
    patch_h = height // 14
    patch_w = width // 14
    cls_pos = self.pos_embed[:, :1]
    patch_pos = self.pos_embed[:, 1:]
    source_size = int(math.sqrt(patch_pos.shape[1]))
    patch_pos = patch_pos.reshape(1, source_size, source_size, 384).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos,
        size=(patch_h, patch_w),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, patch_h * patch_w, 384)
    return torch.cat([cls_pos, patch_pos], dim=1)

  def layer_mix_weights(self) -> torch.Tensor:
    if self.mix_logits is None:
      return torch.empty(0, device=self.reducer.weight.device)
    return self.mix_logits.softmax(dim=0)

  def layer_mix_metrics(self) -> dict[str, float]:
    if self.mix_logits is None:
      return {}
    weights = self.layer_mix_weights().detach().cpu()
    return {
        f"dinov2_layer_mix/block_{block:02d}": float(weight)
        for block, weight in zip(self.mix_blocks, weights, strict=True)
    }

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    height, width = x.shape[-2:]
    x = self.input_adapter(x)
    x = (x - self.input_mean) / self.input_std
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = x + self._position_embedding(height, width).to(dtype=x.dtype, device=x.device)
    if self.output_mode == "layer_mix":
      block_outputs = {}
      mix_block_set = set(self.mix_blocks)
      for block_index, block in enumerate(self.blocks, start=1):
        x = block(x)
        if block_index in mix_block_set:
          block_outputs[block_index] = self.norm(x)[:, 1:]
      stacked_outputs = torch.stack(
          [block_outputs[block_index] for block_index in self.mix_blocks],
          dim=0,
      )
      mix_weights = self.layer_mix_weights().to(
          dtype=stacked_outputs.dtype,
          device=stacked_outputs.device,
      ).view(-1, 1, 1, 1)
      patch_sequence = (stacked_outputs * mix_weights).sum(dim=0)
    else:
      for block in self.blocks[:self.output_block]:
        x = block(x)
      patch_sequence = self.norm(x)[:, 1:]
    patch_h = height // 14
    patch_w = width // 14
    patch_tokens = patch_sequence.transpose(1, 2).reshape(x.shape[0], 384, patch_h, patch_w)
    return self.reducer(patch_tokens)


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
      initializer: InitializerName = "xavier_uniform",
  ) -> None:
    super().__init__()
    resolved_weights = self._resolve_weights(weights)
    backbone = resnet18(weights=resolved_weights)
    if resolved_weights is None:
      mean = torch.zeros(backbone.conv1.in_channels)
      std = torch.ones(backbone.conv1.in_channels)
    else:
      image_mean = torch.as_tensor(resolved_weights.transforms().mean)
      image_std = torch.as_tensor(resolved_weights.transforms().std)
      mean = image_mean
      std = image_std
    self.input_adapter = self._make_input_adapter(input_channels, backbone.conv1.in_channels)
    self.register_buffer("input_mean", mean.view(1, backbone.conv1.in_channels, 1, 1))
    self.register_buffer("input_std", std.view(1, backbone.conv1.in_channels, 1, 1))
    self.stem = nn.Sequential(
        backbone.conv1,
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
    elif variant == "resnet_layer1_reduced":
      self.layers = nn.Sequential(backbone.layer1)
      self.reducer = nn.Conv2d(64, 16, kernel_size=1)
      self.output_channels = 16
    elif variant == "resnet_layer2_reduced":
      self.layers = nn.Sequential(
          backbone.layer1,
          backbone.layer2,
      )
      self.reducer = nn.Conv2d(128, 32, kernel_size=1)
      self.output_channels = 32
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
    elif variant == "resnet_layer4_reduced":
      self.layers = nn.Sequential(
          backbone.layer1,
          backbone.layer2,
          backbone.layer3,
          backbone.layer4,
      )
      self.reducer = nn.Conv2d(512, 128, kernel_size=1)
      self.output_channels = 128
    else:
      raise ValueError(f"Unsupported resnet18 variant {variant!r}.")
    self.variant = variant
    if self.reducer is not None:
      apply_initializer(self.reducer, initializer)

  def _resolve_weights(self, weights: str | None) -> ResNet18_Weights | None:
    if weights is None or str(weights).lower() in {"", "none", "false"}:
      return None
    if str(weights).lower() in {"default", "imagenet", "imagenet1k"}:
      return ResNet18_Weights.DEFAULT
    return ResNet18_Weights[weights]

  def _make_input_adapter(self, input_channels: int, output_channels: int) -> nn.Conv2d:
    adapter = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    with torch.no_grad():
      adapter.weight.fill_(1.0 / input_channels)
      adapter.bias.zero_()
    return adapter

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_adapter(x)
    x = (x - self.input_mean) / self.input_std
    x = self.layers(self.stem(x))
    if self.reducer is not None:
      x = self.reducer(x)
    return x
