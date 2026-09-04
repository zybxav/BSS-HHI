"""Metrics shared by all experiment drivers."""

from collections import Counter
from typing import Dict, Iterable, Sequence

import numpy as np


def top_k_items(values: Iterable[int], top_k: int) -> list:
    """Return a deterministic top-k: descending count, then ascending item ID."""
    counts = Counter(map(int, values))
    return [
        item
        for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[
            :top_k
        ]
    ]


def true_counts(values: Iterable[int], query_items: Sequence[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64)
    counts = np.bincount(arr, minlength=max(int(arr.max()) + 1, 1))
    return np.asarray([counts[x] if x < counts.size else 0 for x in query_items], dtype=float)


def fo_metrics(truth: Sequence[float], estimates: Sequence[float]) -> Dict[str, float]:
    t = np.asarray(truth, dtype=float)
    e = np.asarray(estimates, dtype=float)
    if t.shape != e.shape:
        raise ValueError("truth and estimates must have the same shape")
    error = e - t
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def hhi_metrics(
    truth_items: Sequence[int], estimated_items: Sequence[int], top_k: int
) -> Dict[str, float]:
    """Compute strict precision/recall/F1 at k.

    Missing output slots count as misses.  This prevents a method returning,
    for example, four correct items from receiving precision 1.0 in a top-32
    experiment.
    """
    truth = set(map(int, truth_items[:top_k]))
    estimate = set(map(int, estimated_items[:top_k]))
    hits = len(truth & estimate)
    precision = hits / max(1, top_k)
    recall = hits / max(1, len(truth))
    return {
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "hits": hits,
    }
