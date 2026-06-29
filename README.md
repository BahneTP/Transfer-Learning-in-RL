# Master Thesis Repository

This repository contains the code for my master thesis experiments on model-free reinforcement learning from pixel observations.

It builds on the [TorchRL Hydra Template](https://github.com/raphaelschwinger/torchrl-hydra-template#contribution), which provides the general project structure, Hydra-based configuration management, and a modular TorchRL training pipeline.

## Codebases

The reinforcement learning implementations used in this project are adapted from the following repositories:

| Component | Source |
|----------|--------|
| SAC-BBF | [SAC-BBF-pytorch](https://github.com/BahneTP/SAC-BBF-pytorch/tree/945afa88d3dd5ceaadce6f2a61a897288912153f) |
| Rainbow | [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow) |
| Atari 100K DER/SPR/SR-SPR/BBF/SAC-BBF | Local PyTorch port in `BBF-pytorch` |

## Atari 100K Experiments

The framework includes Atari 100K ports for DER, SPR, SR-SPR, BBF, and SAC-BBF under
`src/algorithms/atari100k`.

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

## Atari 100K Transfer Learning

DER, SPR, SR-SPR, BBF, and SAC-BBF expose optional encoder transfer-learning
knobs on the algorithm config. The default `transfer_mode: none` keeps the
original random-initialized Atari 100K agents.

ResNet-18 can be selected as an encoder while keeping the existing Atari replay,
target, C51, and SPR code paths:

```shell
python src/train.py experiment=atari100k/der/qbert algorithm.encoder_type=resnet18
python src/train.py experiment=atari100k/der/qbert algorithm.encoder_type=resnet18 algorithm.resnet18_weights=DEFAULT
```

Transfer comparison modes:

- `transfer_mode=full_finetune`: encoder, projection/probe, and heads train.
  Use `encoder_lr_scale` to give the encoder a smaller learning rate.
- `transfer_mode=linear_probe`: encoder is frozen; the flat projection and heads
  train.
- `transfer_mode=attentive_probe`: encoder is frozen; a trainable attention
  pooling probe over spatial encoder features and the heads train.
- `transfer_mode=lora`: encoder base weights are frozen; trainable low-rank
  LoRA adapters are added to encoder convolution/linear layers, and the
  projection/probe plus heads train.

Example full fine-tuning run with a smaller encoder learning rate:

```shell
python src/train.py experiment=atari100k/der/qbert \
  algorithm.encoder_type=resnet18 \
  algorithm.resnet18_weights=DEFAULT \
  algorithm.transfer_mode=full_finetune \
  algorithm.encoder_lr_scale=0.1 \
  algorithm.freeze_encoder_bn=true
```

Example LoRA run:

```shell
python src/train.py experiment=atari100k/der/qbert \
  algorithm.encoder_type=resnet18 \
  algorithm.resnet18_weights=DEFAULT \
  algorithm.transfer_mode=lora \
  algorithm.lora_rank=8 \
  algorithm.lora_alpha=16.0
```

For BBF transfer runs, set `algorithm.protect_encoder_from_reset=true` to keep
the periodic reset/shrink-perturb machinery from perturbing the transferred
encoder.

## Logging note

For Atari training runs that learn on clipped rewards, the training environment
can preserve a second, unclipped reward track for logging. In that setup:

- `train/raw_reward` reports the raw training score
- `train/clip_reward` reports the clipped reward used for learning

Available Atari environment configs now include:

- `pong_train` / `pong_eval`
- `qbert_train` / `qbert_eval`
- `battlezone_train` / `battlezone_eval`

## Reproducibility

`trainer.seed` seeds Python, NumPy, PyTorch, the training environment, the
evaluation environment, and parallel environment workers. Training and
evaluation use separate deterministic seed streams, so they do not share an
environment seed.

The default `trainer.deterministic: false` keeps the normal high-performance
PyTorch kernels. Set it to `true` for debugging or strict reproducibility; this
requires deterministic PyTorch operations and disables cuDNN benchmarking, so
it may reduce performance or raise an error when an operation has no
deterministic implementation.

Exact continuation from a mid-episode checkpoint is not guaranteed because
live environment state is not part of algorithm checkpoints.
