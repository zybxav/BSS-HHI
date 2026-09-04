"""Deterministic, scalable wrapper over pure-LDP's Apple CMS/HCMS.

The public xxhash functions and server reconstruction are retained.  For large
datasets we sample aggregate sufficient statistics instead of allocating one
Python report object (or a width-sized vector) per user.
"""

import math
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import xxhash

from reproductions.common.interfaces import FrequencyOracle, validate_inputs, validate_queries
from reproductions.fo.hadaoracle.oracle import fwht, hadamard_sign


class AppleCMS(FrequencyOracle):
    def __init__(self, hadamard: bool = False, rows: int = 32, width: int = 1024):
        self.hadamard = bool(hadamard)
        self.rows = int(rows)
        self.width = int(width)
        self.name = (
            "Apple HCMS (pure-LDP)"
            if self.hadamard
            else "Apple CMS (pure-LDP)"
        )
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
        if self.hadamard and (self.width <= 0 or self.width & (self.width - 1)):
            raise ValueError("HCMS width must be a positive power of two")
        rng = np.random.default_rng(int(seed))
        started = time.perf_counter()
        row_assignment = rng.integers(0, self.rows, size=arr.size, dtype=np.int64)
        # Only distinct observed/query items are hashed in Python.  This keeps
        # pure-LDP's exact public hash family while avoiding n*k Python calls.
        needed = np.unique(np.concatenate((arr, queries)))
        inverse = np.full(domain_size, -1, dtype=np.int64)
        inverse[needed] = np.arange(needed.size)
        hashed = np.empty((self.rows, needed.size), dtype=np.int64)
        for row in range(self.rows):
            hashed[row] = [
                xxhash.xxh64(str(int(item)), seed=row).intdigest() % self.width
                for item in needed
            ]

        sketch = np.zeros((self.rows, self.width), dtype=float)
        if self.hadamard:
            flip_probability = 1.0 / (1.0 + math.exp(epsilon))
            c = (math.exp(epsilon) + 1.0) / (math.exp(epsilon) - 1.0)
            columns = rng.integers(0, self.width, size=arr.size, dtype=np.int64)
            true_buckets = hashed[row_assignment, inverse[arr]]
            signs = hadamard_sign(
                columns.astype(np.uint64), true_buckets.astype(np.uint64)
            )
            signs[rng.random(arr.size) < flip_probability] *= -1.0
            flat = row_assignment * self.width + columns
            sketch = (
                np.bincount(
                    flat, weights=self.rows * c * signs, minlength=self.rows * self.width
                )
                .reshape(self.rows, self.width)
                .astype(float)
            )
        else:
            flip_probability = 1.0 / (1.0 + math.exp(epsilon / 2.0))
            c = (math.exp(epsilon / 2.0) + 1.0) / (
                math.exp(epsilon / 2.0) - 1.0
            )
            # Conditional on a public row, each sketch coordinate is a sum of
            # independent randomized signs. Sample these binomial totals exactly.
            for row in range(self.rows):
                local = arr[row_assignment == row]
                local_buckets = hashed[row, inverse[local]]
                true_counts = np.bincount(local_buckets, minlength=self.width)
                false_counts = local.size - true_counts
                positive_true = rng.binomial(true_counts, 1.0 - flip_probability)
                positive_false = rng.binomial(false_counts, flip_probability)
                sign_sums = (
                    2.0 * (positive_true + positive_false) - local.size
                )
                sketch[row] = self.rows * (
                    (c / 2.0) * sign_sums + 0.5 * local.size
                )
        client_time = time.perf_counter() - started

        server_started = time.perf_counter()
        reconstructed = (
            np.vstack([fwht(row) for row in sketch]) if self.hadamard else sketch
        )
        query_hashes = hashed[:, inverse[queries]]
        column_indices = np.arange(queries.size)
        sums = reconstructed[np.arange(self.rows)[:, None], query_hashes].sum(axis=0)
        estimates = (self.width / (self.width - 1.0)) * (
            sums / self.rows - arr.size / self.width
        )
        server_time = time.perf_counter() - server_started
        bits = (
            1 + math.ceil(math.log2(self.rows)) + math.ceil(math.log2(self.width))
            if self.hadamard
            else self.width + math.ceil(math.log2(self.rows))
        )
        self.last_diagnostics = {
            "rows": self.rows,
            "width": self.width,
            "parameter_source": (
                "pure-LDP group5_apple_vary_eps: k=32, m=1024"
            ),
            "hadamard": self.hadamard,
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": bits,
            "vendor_commit": "d4b2827b49127119b710271e48ac59710872b531",
            "aggregation": "exact aggregate sufficient statistics; pure-LDP public hashes and reconstruction",
        }
        return np.asarray(estimates, dtype=float)

    def parameters(self) -> Dict[str, object]:
        return dict(self.last_diagnostics)
