# Atari 100K: DER, SPR, BBF

This package ports the Atari 100K agents from `BBF-pytorch` into the
TorchRL/Hydra framework.

Implemented algorithms:

| Algorithm | Config |
|-----------|--------|
| DER | `algorithm=atari100k_der` |
| SPR | `algorithm=atari100k_spr` |
| BBF | `algorithm=atari100k_bbf` |

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
python src/train.py experiment=atari100k/bbf/qbert
python src/train.py experiment=atari100k/bbf/battlezone
```

## Known Framework Differences From `BBF-pytorch`

- Collection/evaluation are driven by the framework `StepTrainer` and TorchRL
  collector instead of the standalone `Runner`.
- Policies read and write TorchRL `TensorDict`s.
- Atari preprocessing is expressed as Hydra/TorchRL environment transforms.
- Logging and checkpointing use framework callbacks.

The algorithmic parts that most directly affect learning are kept hard-ported.
