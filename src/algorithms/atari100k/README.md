# Atari 100K Algorithms

This package contains Atari 100K agents ported from the standalone PyTorch BBF
implementation used in this thesis project.

## DER

`DERAtari100KAlgorithm` implements Data-Efficient Rainbow with:

- C51 categorical value distribution with 51 atoms.
- Double DQN targets.
- Dueling Q-head.
- Noisy linear layers.
- Prioritized replay.
- 10-step returns.
- Atari 100K warm-up and epsilon schedule.

The algorithm owns its replay buffer, network, exploration schedule, target
network updates, and collector settings. The trainer only provides batches from
TorchRL's collector and handles logging/checkpointing.

## Atari 100K Environment Assumptions

The Atari 100K configs use:

- `repeat_action_probability: 0.0` for no sticky actions.
- `NoopResetEnv(noops=30, random=True)`.
- `EndOfLifeTransform` during training.
- No `EndOfLifeTransform` during evaluation.
- No reward clipping in the Atari 100K DER/BBF-style configs.
- No `VecNorm`, to keep pixel observations aligned with the standalone port.

## Current Status

Implemented:

- `DERAtari100KAlgorithm`
- `SPRAtari100KAlgorithm`

Available experiment configs:

- `experiment=der/qbert_atari100k`
- `experiment=der/battlezone_atari100k`
- `experiment=spr/qbert_atari100k`
- `experiment=spr/battlezone_atari100k`

The batch script `scripts/run_atari100k_der_spr_batch.sh` runs DER and SPR on
Qbert and BattleZone for seeds 1-3, evaluates each final checkpoint, and prints
a compact results table at the end.

Planned follow-up branches:

- SR-SPR
- BBF
- SAC-BBF
