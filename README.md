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
python src/train.py experiment=atari100k/dummy/qbert
python src/train.py experiment=atari100k/dummy/battlezone
```

## Logging note

For Atari training runs that learn on clipped rewards, the training environment
can preserve a second, unclipped reward track for logging. In that setup:

- `train/raw_reward` reports the raw training score
- `train/clip_reward` reports the clipped reward used for learning

Available Atari environment configs now include:

- `pong_train` / `pong_eval`
- `qbert_train` / `qbert_eval`
- `battlezone_train` / `battlezone_eval`
