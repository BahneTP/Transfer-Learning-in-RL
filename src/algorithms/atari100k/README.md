# Atari 100K: DER, SPR, SR-SPR, BBF, SAC-BBF

This package ports the Atari 100K agents from `BBF-pytorch` into the
TorchRL/Hydra framework.

Implemented algorithms:

| Algorithm | Config |
|-----------|--------|
| DER | `algorithm=atari100k_der` |
| SPR | `algorithm=atari100k_spr` |
| SR-SPR | `algorithm=atari100k_sr_spr` |
| BBF | `algorithm=atari100k_bbf` |
| SAC-BBF | `algorithm=atari100k_sac_bbf` |

The implementation keeps the learning core close to `BBF-pytorch`: C51
distributional targets, n-step returns, deterministic prioritized replay,
NoisyNet layers, dueling heads, SPR rollouts, and BBF reset logic are local to
this package. The framework adapter translates TorchRL `TensorDict` batches into
the NumPy replay format used by those agents.

## Atari 100K Environment

Experiments use:

- `configs/environment/atari100k_train.yaml`
- `configs/environment/atari100k_eval.yaml`

The environment emits single 84x84 grayscale frames. Frame stacking stays inside
the Atari100K algorithm/replay, matching `BBF-pytorch`.

## Experiments

```shell
python src/train.py experiment=atari100k/der/assault
python src/train.py experiment=atari100k/der/bankheist
python src/train.py experiment=atari100k/der/roadrunner
python src/train.py experiment=atari100k/der/breakout
python src/train.py experiment=atari100k/der/hero
python src/train.py experiment=atari100k/der/jamesbond
python src/train.py experiment=atari100k/bbf/assault
python src/train.py experiment=atari100k/bbf/bankheist
python src/train.py experiment=atari100k/bbf/roadrunner
python src/train.py experiment=atari100k/bbf/breakout
python src/train.py experiment=atari100k/bbf/hero
python src/train.py experiment=atari100k/bbf/jamesbond
```

## Known Framework Differences From `BBF-pytorch`

- Collection/evaluation are driven by the framework `StepTrainer` and TorchRL
  collector instead of the standalone `Runner`.
- Policies read and write TorchRL `TensorDict`s.
- Atari preprocessing is expressed as config-composed gymnasium wrappers plus
  TorchRL environment transforms.
- Logging and checkpointing use framework callbacks.

The algorithmic parts that most directly affect learning are kept hard-ported.

## Transfer Learning

Transfer learning is implemented as an optional extension to the shared Atari
100K network stack. The default configs keep `transfer_mode: none`, so baseline
DER/SPR/SR-SPR/BBF/SAC-BBF runs remain random-initialized ports of the original
agents. Transfer-learning experiment configs are provided for DER and SAC-BBF.

Available encoder choices:

| Encoder | `encoder_type` | Notes |
|---------|----------------|-------|
| Nature/Rainbow CNN | `dqn` | DER/SPR-scale default |
| IMPALA CNN | `impala` | BBF/SAC-BBF default |
| ResNet-18 trunk | `resnet18` | Torchvision ResNet-18 without average pool/classifier |

For `encoder_type=resnet18`, Atari frame stacks have shape `(B, 4, 84, 84)`.
The first ResNet convolution is adapted from RGB to four grayscale frame
channels. With `resnet18_weights=DEFAULT`, torchvision ImageNet weights are
loaded and the encoder applies the corresponding normalization averaged across
the grayscale channels.

ResNet-18 has three explicit feature variants:

```text
resnet_full             -> layer4 output: 512x3x3
resnet_layer3_flattened -> layer3 output: 256x6x6
resnet_layer3_reduced   -> layer3 output + trainable 1x1 reducer: 64x6x6
```

Projection/probing modes:

- `transfer_mode=full_finetune`: encoder, projection/probe, transition model,
  and heads train. `encoder_lr_scale` multiplies the base learning rate for
  encoder parameters.
- `transfer_mode=linear_probe`: encoder is frozen. The existing flat projection
  maps spatial ResNet features to `hidden_dim`, and the heads train.
- `transfer_mode=attentive_probe`: encoder is frozen. A small trainable
  attention pooling probe scores spatial ResNet tokens and maps the pooled
  feature to `hidden_dim`; the heads train.
- `transfer_mode=lora`: encoder base weights are frozen. Low-rank LoRA adapters
  are inserted into encoder `Conv2d`/`Linear` layers; only those adapter weights,
  the projection/probe, transition model, and heads train.

Use `freeze_encoder_bn=true` for pretrained ResNet runs when BatchNorm running
statistics should stay fixed. For SAC-BBF transfer experiments, use
`protect_encoder_from_reset=true` to keep periodic reset/shrink-perturb from
modifying the transferred encoder while still allowing the transition model and
heads to reset according to the BBF config.

The BBF baseline keeps the original reset/shrink-perturb behavior. SAC-BBF
transfer experiment YAMLs set `protect_encoder_from_reset=true`, so the
transferred encoder is protected while the rest of the reset policy remains active.

Transfer settings are stored in the resolved Hydra config and, when enabled,
the W&B run config. They are not duplicated as `train/*` metrics.

Example DER full fine-tuning run:

```shell
python src/train.py experiment=atari100k/der/assault \
  algorithm.encoder_type=resnet18 \
  algorithm.resnet18_weights=DEFAULT \
  algorithm.transfer_mode=full_finetune \
  algorithm.encoder_lr_scale=0.1 \
  algorithm.freeze_encoder_bn=true
```

Example DER LoRA run:

```shell
python src/train.py experiment=atari100k/der/assault \
  algorithm.encoder_type=resnet18 \
  algorithm.resnet18_weights=DEFAULT \
  algorithm.transfer_mode=lora \
  algorithm.lora_rank=4 \
  algorithm.lora_alpha=8.0
```
