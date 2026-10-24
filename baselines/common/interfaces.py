"""Small common interfaces and result helpers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


def validate_inputs(
    values: Iterable[int], domain_size: int, epsilon: float
) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not isinstance(domain_size, (int, np.integer)) or domain_size < 2:
        raise ValueError("domain_size must be an integer >= 2")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0")
    if np.any(arr < 0) or np.any(arr >= domain_size):
        raise ValueError("values contain IDs outside [0, domain_size)")
    return arr


class FrequencyOracle(ABC):
    name = "FrequencyOracle"

    @abstractmethod
    def estimate(
        self,
        values: Iterable[int],
        domain_size: int,
        epsilon: float,
        seed: int,
        query_items: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return estimated counts, ordered by query_items or the whole domain."""

    def parameters(self) -> Dict[str, Any]:
        return {}


class HeavyHitter(ABC):
    name = "HeavyHitter"

    @abstractmethod
    def run(
        self,
        values: Iterable[int],
        domain_size: int,
        epsilon: float,
        top_k: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Run candidate discovery and return the common result dictionary."""


def validate_queries(
    query_items: Optional[Sequence[int]], domain_size: int
) -> np.ndarray:
    q = (
        np.arange(domain_size, dtype=np.int64)
        if query_items is None
        else np.asarray(query_items, dtype=np.int64)
    )
    if q.ndim != 1 or np.any(q < 0) or np.any(q >= domain_size):
        raise ValueError("query_items must lie in [0, domain_size)")
    return q

