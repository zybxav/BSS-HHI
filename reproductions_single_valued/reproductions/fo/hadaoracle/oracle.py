"""Hadamard randomized response with FWHT aggregation.

Each user reports one public column and one randomized sign.  Orthogonality of
Hadamard rows makes ``H @ column_sums / alpha`` an unbiased count estimator.
"""

import math
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from reproductions.common.interfaces import (
    FrequencyOracle,
    validate_inputs,
    validate_queries,
)


def next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def fwht(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if out.ndim != 1 or out.size == 0 or out.size & (out.size - 1):
        raise ValueError("FWHT input length must be a positive power of two")
    step = 1
    while step < out.size:
        view = out.reshape(-1, 2 * step)
        left = view[:, :step].copy()
        right = view[:, step : 2 * step].copy()
        view[:, :step] = left + right
        view[:, step : 2 * step] = left - right
        step *= 2
    return out


def hadamard_sign(row: np.ndarray, column: np.ndarray) -> np.ndarray:
    parity = np.zeros(np.broadcast(row, column).shape, dtype=np.uint8)
    bits = np.bitwise_and(row, column).astype(np.uint64, copy=False)
    while np.any(bits):
        parity ^= (bits & 1).astype(np.uint8)
        bits >>= 1
    return 1.0 - 2.0 * parity


def _uint64_parity(values: np.ndarray) -> np.ndarray:
    work = np.asarray(values, dtype=np.uint64).copy()
    work ^= work >> np.uint64(32)
    work ^= work >> np.uint64(16)
    work ^= work >> np.uint64(8)
    work ^= work >> np.uint64(4)
    work ^= work >> np.uint64(2)
    work ^= work >> np.uint64(1)
    return (work & np.uint64(1)).astype(np.int64)


def linear_pairwise_hash(
    values: np.ndarray, masks: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Hash integers with a random affine map over GF(2).

    Each output bit is ``<mask, value> xor offset``.  Independent random masks
    and offsets make the output uniform and pairwise independent for every two
    distinct inputs.  The range size is exactly ``2**len(masks)``.
    """
    values = np.asarray(values, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if masks.ndim != 1 or offsets.shape != masks.shape:
        raise ValueError("masks and offsets must be equal-length vectors")
    output = np.zeros(values.size, dtype=np.int64)
    for bit, (mask, offset) in enumerate(zip(masks, offsets)):
        output |= (
            _uint64_parity(values & mask) ^ int(offset)
        ) << bit
    return output


class HadaOracle(FrequencyOracle):
    name = "HadaOracle-core (independent)"

    def __init__(
        self,
        repetitions: Optional[int] = None,
        failure_probability: float = 0.05,
        constant_ck: float = 8.0,
    ):
        self.repetitions = None if repetitions is None else int(repetitions)
        self.failure_probability = float(failure_probability)
        self.constant_ck = float(constant_ck)
        self.last_diagnostics: Dict[str, object] = {}

    def privatize(
        self,
        values: Iterable[int],
        domain_size: int,
        epsilon: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        arr = validate_inputs(values, domain_size, epsilon)
        size = next_power_of_two(domain_size)
        rng = np.random.default_rng(int(seed))
        columns = rng.integers(0, size, size=arr.size, dtype=np.int64)
        signs = hadamard_sign(arr.astype(np.uint64), columns.astype(np.uint64))
        keep_probability = math.exp(epsilon) / (math.exp(epsilon) + 1.0)
        flips = rng.random(arr.size) >= keep_probability
        signs[flips] *= -1.0
        return columns, signs.astype(np.int8), size

    def aggregate(
        self, columns: np.ndarray, signs: np.ndarray, size: int, epsilon: float
    ) -> np.ndarray:
        column_sums = np.bincount(columns, weights=signs, minlength=size)
        alpha = math.tanh(epsilon / 2.0)
        return fwht(column_sums) / alpha

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
        if not 0.0 < self.failure_probability < 1.0:
            raise ValueError("failure_probability must be in (0, 1)")
        if self.constant_ck <= 0:
            raise ValueError("constant_ck must be positive")
        paper_rows = int(
            math.ceil(
                self.constant_ck
                * math.log(4.0 / self.failure_probability)
            )
        )
        repetitions = max(
            1,
            min(
                int(self.repetitions or paper_rows),
                arr.size,
            ),
        )
        # Wu-Wirth Algorithm 1 initializes
        #   k = C_K ln(4/beta_0)
        #   m = 8 e^2 sqrt(C_K) epsilon sqrt(|U|).
        # HRR requires a power-of-two table, so m is rounded upward and clipped
        # when the original domain is already smaller.
        target_width = max(
            2,
            int(
                math.ceil(
                    8.0
                    * math.e**2
                    * math.sqrt(self.constant_ck)
                    * epsilon
                    * math.sqrt(arr.size)
                )
            ),
        )
        sketch_width = min(next_power_of_two(domain_size), next_power_of_two(target_width))
        stream = np.random.default_rng(int(seed))
        groups = np.arange(arr.size, dtype=np.int64) % repetitions
        stream.shuffle(groups)
        output_bits = int(math.log2(sketch_width))
        hash_masks = stream.integers(
            0,
            np.iinfo(np.uint64).max,
            size=(repetitions, output_bits),
            dtype=np.uint64,
            endpoint=True,
        )
        hash_offsets = stream.integers(
            0, 2, size=(repetitions, output_bits), dtype=np.int64
        )
        per_repetition = []
        client_time = 0.0
        server_time = 0.0
        for repeat in range(repetitions):
            mask = groups == repeat
            local = arr[mask]
            mapped = (
                linear_pairwise_hash(
                    local, hash_masks[repeat], hash_offsets[repeat]
                )
                if sketch_width < domain_size
                else local
            )
            started = time.perf_counter()
            columns, signs, size = self.privatize(
                mapped, sketch_width, epsilon, int(stream.integers(0, 2**63 - 1))
            )
            client_time += time.perf_counter() - started
            server_started = time.perf_counter()
            table = self.aggregate(columns, signs, size, epsilon)
            hashed_queries = (
                linear_pairwise_hash(
                    queries, hash_masks[repeat], hash_offsets[repeat]
                )
                if sketch_width < domain_size
                else queries
            )
            local_estimates = table[hashed_queries]
            # Algorithm 1 scales the base-oracle estimate of the hashed
            # frequency by k and takes a median. Hash-collision error is handled
            # by the median analysis; it is not explicitly subtracted.
            per_repetition.append(local_estimates * repetitions)
            server_time += time.perf_counter() - server_started
        estimates = np.median(np.vstack(per_repetition), axis=0)
        self.last_diagnostics = {
            "hadamard_size": sketch_width,
            "sketch_width": sketch_width,
            "repetitions": repetitions,
            "paper_rows": paper_rows,
            "failure_probability_beta0": self.failure_probability,
            "constant_CK": self.constant_ck,
            "paper_target_width": target_width,
            "hash_family": (
                "exact pairwise-independent affine GF(2) map to the "
                "power-of-two sketch domain"
            ),
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": 1,
            "alpha": math.tanh(epsilon / 2.0),
            "report_semantics": "one randomized sign; group, hash, and Hadamard column are public coins",
            "aggregation": (
                "Wu-Wirth Algorithm 1: scaled median of disjoint hashed "
                "HRR rows with FWHT"
            ),
        }
        return estimates

    def parameters(self) -> Dict[str, object]:
        return dict(self.last_diagnostics)
