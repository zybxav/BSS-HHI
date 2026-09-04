"""Deterministic seed derivation without mutating process-global RNG state."""

import hashlib
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np


class SeedStream:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def derive(self, label: str) -> int:
        payload = f"{self.seed}:{label}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

    def rng(self, label: str = "default") -> np.random.Generator:
        return np.random.default_rng(self.derive(label))


@contextmanager
def legacy_seed(seed: int) -> Iterator[None]:
    """Temporarily seed libraries that use global random state (pure-LDP)."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

