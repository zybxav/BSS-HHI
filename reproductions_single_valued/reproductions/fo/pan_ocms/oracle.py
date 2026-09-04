"""Pan-OCMS core from arXiv:2603.11523, Section 5.2.

The hash family is h_{a,b}(x)=((a*x+b) mod D') mod m where D' is prime,
a is non-zero, and b is arbitrary.  The exact collision probability of this
family is used in the count-mean correction.
"""

import math
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from reproductions.common.interfaces import FrequencyOracle, validate_inputs, validate_queries


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 2
    return True


def next_prime(value: int) -> int:
    candidate = max(2, int(value))
    while not is_prime(candidate):
        candidate += 1
    return candidate


class PanOCMS(FrequencyOracle):
    name = "Pan-OCMS (independent)"

    def __init__(self, buckets: Optional[int] = None, cohort_rows: Optional[int] = None):
        self.buckets = buckets
        self.cohort_rows = None if cohort_rows is None else int(cohort_rows)
        self.last_diagnostics: Dict[str, object] = {}

    @staticmethod
    def optimized_buckets(epsilon: float) -> int:
        # Pan 2026, Section 5.2: B = round(1 + exp(epsilon)).
        return max(2, int(math.floor(math.exp(epsilon) + 1.5)))

    @staticmethod
    def collision_probability(prime_domain: int, buckets: int) -> float:
        quotient, remainder = divmod(prime_domain, buckets)
        ordered_collisions = (
            remainder * (quotient + 1) * quotient
            + (buckets - remainder) * quotient * (quotient - 1)
        )
        return float(
            ordered_collisions / (prime_domain * (prime_domain - 1))
        )

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
        prime = next_prime(domain_size)
        buckets = min(prime, int(self.buckets or self.optimized_buckets(epsilon)))
        if buckets < 2:
            raise ValueError("buckets must be >= 2")
        rng = np.random.default_rng(int(seed))
        started = time.perf_counter()
        rows = (
            arr.size
            if self.cohort_rows is None
            else max(1, min(self.cohort_rows, arr.size))
        )
        groups = np.arange(arr.size, dtype=np.int64) % rows
        rng.shuffle(groups)
        a = rng.integers(1, prime, size=rows, dtype=np.int64)
        b = rng.integers(0, prime, size=rows, dtype=np.int64)
        true_bucket = ((a[groups] * arr + b[groups]) % prime) % buckets
        exp_eps = math.exp(epsilon)
        p = exp_eps / (exp_eps + buckets - 1.0)
        keep = rng.random(arr.size) < p
        offsets = rng.integers(1, buckets, size=arr.size, dtype=np.int64)
        reports = np.where(keep, true_bucket, (true_bucket + offsets) % buckets)
        report_counts = np.bincount(
            groups * buckets + reports, minlength=rows * buckets
        ).reshape(rows, buckets)
        client_time = time.perf_counter() - started

        server_started = time.perf_counter()
        collision = self.collision_probability(prime, buckets)
        q = 1.0 / (exp_eps + buckets - 1.0)
        background = q + collision * (p - q)
        denominator = p - background
        estimates = np.empty(queries.size, dtype=float)
        # Chunk queries to keep peak memory bounded for large user populations.
        chunk_size = max(1, min(4096, int(48_000_000 // max(1, rows * 24))))
        for start in range(0, queries.size, chunk_size):
            chunk = queries[start : start + chunk_size]
            hashed = ((a[:, None] * chunk[None, :] + b[:, None]) % prime) % buckets
            matches = report_counts[np.arange(rows)[:, None], hashed].sum(axis=0)
            estimates[start : start + chunk.size] = (matches - arr.size * background) / denominator
        server_time = time.perf_counter() - server_started
        self.last_diagnostics = {
            "prime_domain": prime,
            "buckets": buckets,
            "collision_probability": collision,
            "p": p,
            "q": q,
            "background_probability": background,
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": math.ceil(math.log2(buckets)),
            "communication_model": (
                "one perturbed bucket; the per-user affine hash is derived "
                "from public coins and is not transmitted"
            ),
            "hash_family": "affine mod prime with a != 0, then mod buckets",
            "query_chunk_size": chunk_size,
            "public_hash_cohorts": rows,
            "independent_hash_per_user": rows == arr.size,
            "full_domain_scalability": (
                "exact per-user public affine hash" if rows == arr.size else
                "explicit opt-in balanced shared-hash cohort approximation"
            ),
        }
        return estimates

    def parameters(self) -> Dict[str, object]:
        return dict(self.last_diagnostics)
