"""Load the 1-based single-valued files used by the experiment runner."""

from pathlib import Path

import numpy as np


def load_single_values(path: str) -> np.ndarray:
    values = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            tokens = raw.strip().split()
            if not tokens:
                continue
            if len(tokens) != 1:
                raise ValueError(f"expected one ID at line {line_no}")
            values.append(int(tokens[0]))
    if not values:
        raise ValueError("single-valued dataset is empty")
    normalized = np.asarray(values, dtype=np.int64) - 1
    if np.any(normalized < 0):
        raise ValueError("expected positive 1-based item IDs")
    return normalized
