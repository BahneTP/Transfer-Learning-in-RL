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
python src/train.py experiment=atari100k/der/qbert
python src/train.py experiment=atari100k/der/battlezone
python src/train.py experiment=atari100k/spr/qbert
python src/train.py experiment=atari100k/spr/battlezone
python src/train.py experiment=atari100k/sr_spr/qbert
python src/train.py experiment=atari100k/sr_spr/battlezone
python src/train.py experiment=atari100k/bbf/qbert
python src/train.py experiment=atari100k/bbf/battlezone
python src/train.py experiment=atari100k/sac_bbf/qbert
python src/train.py experiment=atari100k/sac_bbf/battlezone
```

## Known Framework Differences From `BBF-pytorch`

- Collection/evaluation are driven by the framework `StepTrainer` and TorchRL
  collector instead of the standalone `Runner`.
- Policies read and write TorchRL `TensorDict`s.
- Atari preprocessing is expressed as Hydra/TorchRL environment transforms.
- Logging and checkpointing use framework callbacks.

The algorithmic parts that most directly affect learning are kept hard-ported.

## Transfer Learning

Transfer learning is implemented as an optional extension to the shared Atari
100K network stack. The default configs keep `transfer_mode: none`, so baseline
DER/SPR/SR-SPR/BBF/SAC-BBF runs remain random-initialized ports of the original
agents.

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

The current ResNet-18 feature path is:

```text
(B, 4, 84, 84)
  -> ResNet-18 trunk
  -> (B, 512, 3, 3)
```

Projection/probing modes:

- `transfer_mode=full_finetune`: encoder, projection/probe, transition model,
  and heads train. `encoder_lr_scale` multiplies the base learning rate for
  encoder parameters.
- `transfer_mode=linear_probe`: encoder is frozen. The existing flat projection
  maps `512 * 3 * 3` features to `hidden_dim`, and the heads train.
- `transfer_mode=attentive_probe`: encoder is frozen. A small trainable
  attention pooling probe scores the nine spatial ResNet tokens and maps the
  pooled feature to `hidden_dim`; the heads train.

Use `freeze_encoder_bn=true` for pretrained ResNet runs when BatchNorm running
statistics should stay fixed. For BBF-family transfer experiments, use
`protect_encoder_from_reset=true` to keep periodic reset/shrink-perturb from
modifying the transferred encoder while still allowing the transition model and
heads to reset according to the BBF config.

Example DER full fine-tuning run:

```shell
python src/train.py experiment=atari100k/der/qbert \
  algorithm.encoder_type=resnet18 \
  algorithm.resnet18_weights=DEFAULT \
  algorithm.transfer_mode=full_finetune \
  algorithm.encoder_lr_scale=0.1 \
  algorithm.freeze_encoder_bn=true
```
