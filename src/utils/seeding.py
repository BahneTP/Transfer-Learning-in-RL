import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed process-wide RNGs and optionally require deterministic PyTorch ops.

    ``PYTHONHASHSEED`` is exported for child processes. Python's hash seed for
    the current interpreter is fixed at process startup and therefore cannot be
    changed here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False


def derive_seed(seed: int, stream: str) -> int:
    """Derive a stable, independent 32-bit seed for a named RNG stream."""
    payload = f"{int(seed)}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
