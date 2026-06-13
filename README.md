# Master Thesis Repository

This repository contains the code for my master thesis experiments on model-free reinforcement learning from pixel observations.

It builds on the [TorchRL Hydra Template](https://github.com/raphaelschwinger/torchrl-hydra-template#contribution), which provides the general project structure, Hydra-based configuration management, and a modular TorchRL training pipeline.

## Codebases

The reinforcement learning implementations used in this project are adapted from the following repositories:

| Component | Source |
|----------|--------|
| SAC-BBF | [SAC-BBF-pytorch](https://github.com/BahneTP/SAC-BBF-pytorch/tree/945afa88d3dd5ceaadce6f2a61a897288912153f) |
| Rainbow | [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow) |

## Implemented Algorithms

| Algorithm | Environment | Config |
|----------|-------------|--------|
| DER Atari 100K | ALE/Qbert-v5 | `experiment=der/qbert_atari100k` |
| SPR Atari 100K | ALE/Qbert-v5 | `experiment=spr/qbert_atari100k` |
