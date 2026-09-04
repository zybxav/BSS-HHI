"""Finite-scale implementation of Bassily et al. Algorithm 7 Hashtogram."""

import math
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from reproductions.common.interfaces import (
    FrequencyOracle,
    validate_inputs,
    validate_queries,
)
from reproductions.fo.hadaoracle.oracle import fwht, hadamard_sign, next_power_of_two
from reproductions.fo.standard.oracles import (
    _linear_hash_queries,
    _linear_hash_reports,
)


class Hashtogram(FrequencyOracle):
    """Random-sign hashed frequency oracle with disjoint public hash rows.

    ``rows`` and ``width`` expose the constants hidden by the paper's
    asymptotic notation.  The mechanism and unbiased random-sign estimator
    follow Algorithm 7; FWHT is only an exact acceleration of all Z[t,j] sums.
    """

    name = "Hashtogram"

    def __init__(self, rows: int = 5, width: Optional[int] = None):
        self.rows = int(rows)
        self.width = width
        self.last_diagnostics: Dict[str, object] = {}

    def estimate(
        self,
        values: Iterable[int],
        domain_size: int,
        epsilon: float,
        seed: int,
        query_items: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        arr = validate_inputs(values, domain_size, epsilon)
        queries = validate_queries(query_items, domain_size)
        client_started = time.perf_counter()
        rows = max(1, min(self.rows, arr.size))
        requested_width = int(self.width or max(8, math.ceil(math.sqrt(arr.size))))
        width = next_power_of_two(requested_width)
        rng = np.random.default_rng(int(seed))
        groups = np.arange(arr.size, dtype=np.int64) % rows
        rng.shuffle(groups)
        output_bits = int(math.log2(width))
        hash_masks = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=(rows, output_bits),
            endpoint=True,
            dtype=np.uint64,
        )
        hash_offsets = rng.integers(
            0, width, size=rows, dtype=np.int64
        )
        columns = rng.integers(0, width, size=arr.size, dtype=np.int64)
        true_bucket = _linear_hash_reports(
            arr, groups, hash_masks, hash_offsets
        )
        reports = hadamard_sign(
            true_bucket.astype(np.uint64), columns.astype(np.uint64)
        )
        keep_probability = math.exp(epsilon) / (math.exp(epsilon) + 1.0)
        reports[rng.random(arr.size) >= keep_probability] *= -1.0
        client_time = time.perf_counter() - client_started

        server_started = time.perf_counter()
        alpha = math.tanh(epsilon / 2.0)
        tables = np.empty((rows, width), dtype=float)
        for row in range(rows):
            mask = groups == row
            column_sums = np.bincount(
                columns[mask], weights=reports[mask], minlength=width
            )
            tables[row] = fwht(column_sums) / alpha
        row_estimates = np.empty((rows, queries.size), dtype=float)
        mapped_queries = _linear_hash_queries(
            queries, hash_masks, hash_offsets
        )
        for row in range(rows):
            row_estimates[row] = rows * tables[row, mapped_queries[row]]
        estimates = np.median(row_estimates, axis=0)
        server_time = time.perf_counter() - server_started
        self.last_diagnostics = {
            "rows": rows,
            "width": width,
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": 1,
            "alpha": alpha,
            "hash_family": (
                "exact pairwise-independent affine GF(2) map to the "
                "power-of-two bucket domain"
            ),
            "implementation": (
                "Algorithm 7 random-sign hash rows; Hadamard columns/FWHT "
                "exactly accelerate the public pairwise-independent Z matrix"
            ),
            "constant_policy": (
                "finite-scale fixed rows and sqrt(local_n) width replace "
                "the paper's asymptotic constants"
            ),
        }
        return np.asarray(estimates, dtype=float)

    def parameters(self):
        return dict(self.last_diagnostics)
