"""Compact OLH/OUE/SUE baselines used by the unified FO experiment."""

import math
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from baselines.common.interfaces import FrequencyOracle, validate_inputs, validate_queries


def _parity_uint64(values):
    """Vectorized parity for NumPy uint64 arrays."""
    values = np.array(values, dtype=np.uint64, copy=True)
    values ^= values >> np.uint64(32)
    values ^= values >> np.uint64(16)
    values ^= values >> np.uint64(8)
    values ^= values >> np.uint64(4)
    values &= np.uint64(0xF)
    return (
        (np.uint64(0x6996) >> values) & np.uint64(1)
    ).astype(np.int64)


def _linear_hash_reports(values, groups, masks, offsets):
    """Evaluate one independently selected GF(2) affine map per user."""
    values = np.asarray(values, dtype=np.uint64)
    output = np.zeros(values.size, dtype=np.int64)
    for bit in range(masks.shape[1]):
        parity = _parity_uint64(values & masks[groups, bit])
        parity ^= (offsets[groups] >> bit) & 1
        output |= parity << bit
    return output


def _linear_hash_queries(queries, masks, offsets):
    """Evaluate every public GF(2) affine map on a query chunk."""
    queries = np.asarray(queries, dtype=np.uint64)
    output = np.zeros((masks.shape[0], queries.size), dtype=np.int64)
    for bit in range(masks.shape[1]):
        parity = _parity_uint64(
            masks[:, bit, None] & queries[None, :]
        )
        parity ^= ((offsets >> bit) & 1)[:, None]
        output |= parity << bit
    return output


def _is_prime(value):
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


def _next_prime(value):
    candidate = max(2, int(value))
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _affine_collision_probability(prime, buckets):
    quotient, remainder = divmod(prime, buckets)
    ordered_collisions = (
        remainder * (quotient + 1) * quotient
        + (buckets - remainder) * quotient * (quotient - 1)
    )
    return float(ordered_collisions / (prime * (prime - 1)))


class UnaryEncodingOracle(FrequencyOracle):
    def __init__(self, optimized: bool = True):
        self.optimized = optimized
        self.name = "OUE" if optimized else "SUE"
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
        if self.optimized:
            p, q = 0.5, 1.0 / (math.exp(epsilon) + 1.0)
        else:
            p = math.exp(epsilon / 2.0) / (math.exp(epsilon / 2.0) + 1.0)
            q = 1.0 - p
        rng = np.random.default_rng(int(seed))
        started = time.perf_counter()
        counts = np.bincount(arr, minlength=domain_size)
        positive = counts[queries]
        # For every queried unary coordinate, the aggregate sufficient statistic
        # is Binomial(f_x,p)+Binomial(n-f_x,q).  Sampling that statistic is
        # distributionally identical to materializing all n report bits.
        observed = rng.binomial(positive, p) + rng.binomial(arr.size - positive, q)
        output = (observed - arr.size * q) / (p - q)
        elapsed = time.perf_counter() - started
        self.last_diagnostics = {
            "p": p,
            "q": q,
            "runtime_client": elapsed,
            "runtime_server": 0.0,
            "communication_bits_per_user": domain_size,
            "simulation": "exact aggregate sufficient-statistic unary simulation",
        }
        return output

    def parameters(self):
        return dict(self.last_diagnostics)


class OLHOracle(FrequencyOracle):
    name = "OLH"

    def __init__(self, cohort_rows: Optional[int] = None):
        self.cohort_rows = None if cohort_rows is None else int(cohort_rows)
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
        # Standard OLH uses g = round(exp(epsilon)) + 1.  A nonzero affine map
        # over a prime field sends every distinct input pair uniformly to the
        # ordered distinct field pairs.  The subsequent modulo-g collision
        # probability is therefore query-independent and can be corrected
        # exactly, without forcing g to a power of two.
        g = max(2, int(round(math.exp(epsilon))) + 1)
        prime = _next_prime(domain_size)
        rng = np.random.default_rng(int(seed))
        started = time.perf_counter()
        rows = (
            arr.size
            if self.cohort_rows is None
            else max(1, min(self.cohort_rows, arr.size))
        )
        groups = np.arange(arr.size, dtype=np.int64) % rows
        rng.shuffle(groups)
        multipliers = rng.integers(1, prime, size=rows, dtype=np.int64)
        offsets = rng.integers(0, prime, size=rows, dtype=np.int64)
        true_hash = (
            (multipliers[groups] * arr + offsets[groups]) % prime
        ) % g
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        keep = rng.random(arr.size) < p
        offset = rng.integers(1, g, size=arr.size)
        reports = np.where(keep, true_hash, (true_hash + offset) % g)
        report_counts = np.bincount(
            groups * g + reports, minlength=rows * g
        ).reshape(rows, g)
        client_time = time.perf_counter() - started
        collision = _affine_collision_probability(prime, g)
        q = 1.0 / (math.exp(epsilon) + g - 1)
        background = q + collision * (p - q)
        server_started = time.perf_counter()
        estimates = np.empty(queries.size, dtype=float)
        chunk_size = max(1, min(4096, int(48_000_000 // max(1, rows * 24))))
        for start in range(0, queries.size, chunk_size):
            chunk = queries[start : start + chunk_size]
            hashed = (
                (
                    multipliers[:, None] * chunk[None, :]
                    + offsets[:, None]
                )
                % prime
            ) % g
            matches = report_counts[np.arange(rows)[:, None], hashed].sum(axis=0)
            estimates[start : start + chunk.size] = (
                matches - arr.size * background
            ) / (p - background)
        server_time = time.perf_counter() - server_started
        self.last_diagnostics = {
            "hash_range": g,
            "prime_domain": prime,
            "collision_probability": float(collision),
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": math.ceil(math.log2(g)),
            "public_hash": (
                "nonzero affine map over a prime field followed by modulo g; "
                "exact query-independent collision correction"
            ),
            "query_chunk_size": chunk_size,
            "public_hash_cohorts": rows,
            "independent_hash_per_user": rows == arr.size,
            "full_domain_scalability": (
                "exact independent public hash per user" if rows == arr.size else
                "explicit opt-in balanced shared-hash cohort approximation"
            ),
        }
        return estimates

    def parameters(self):
        return dict(self.last_diagnostics)
