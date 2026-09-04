"""Finite-scale PrivateExpanderSketch reproduction.

Section 3.3 of Bun--Nelson--Stemmer prescribes full-domain Hashtogram queries
when n > |X|.  For n <= |X| this implementation follows the displayed
PrivateExpanderSketch protocol: one public partition, an outer hash ``g``,
independent coordinate hashes ``h_m``, private frequency estimates of
``(g(x), h_m(x), Enc(x)_m)``, bounded per-bucket lists, list recovery, and an
independent final Hashtogram verification.  A finite-field Reed--Solomon code
is used as the executable unique-list-recovery code because the paper's
asymptotic construction has no released implementation.
"""

import math
import time
from collections import Counter

import numpy as np

from reproductions.common.interfaces import HeavyHitter, validate_inputs
from reproductions.common.randomness import SeedStream
from reproductions.fo.hashtogram import Hashtogram
from reproductions.fo.pan_ocms.oracle import next_prime


FIELD = 257


def _modular_inverse(matrix, prime=FIELD):
    matrix = np.asarray(matrix, dtype=np.int64) % prime
    size = matrix.shape[0]
    augmented = np.concatenate(
        [matrix, np.eye(size, dtype=np.int64)], axis=1
    )
    for column in range(size):
        pivot = next(
            row
            for row in range(column, size)
            if augmented[row, column] % prime
        )
        augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] = (
            augmented[column]
            * pow(int(augmented[column, column]), prime - 2, prime)
        ) % prime
        for row in range(size):
            if row != column:
                augmented[row] = (
                    augmented[row]
                    - augmented[row, column] * augmented[column]
                ) % prime
    return augmented[:, size:]


def _base256_coefficients(values, message_symbols):
    values = np.asarray(values, dtype=np.int64)
    coefficients = np.empty((values.size, message_symbols), dtype=np.int64)
    work = values.copy()
    for position in range(message_symbols):
        coefficients[:, position] = work % 256
        work //= 256
    return coefficients


def _rs_symbols(values, message_symbols, code_positions=None):
    """Evaluate base-256 message polynomials at public field points."""
    coefficients = _base256_coefficients(values, message_symbols)
    positions = int(code_positions or 2 * message_symbols)
    points = np.arange(1, positions + 1, dtype=np.int64)
    powers = np.empty((message_symbols, points.size), dtype=np.int64)
    powers[0] = 1
    for degree in range(1, message_symbols):
        powers[degree] = powers[degree - 1] * points % FIELD
    return coefficients.dot(powers) % FIELD


def _coefficients_to_item(coefficients):
    item = 0
    multiplier = 1
    for coefficient in coefficients:
        if int(coefficient) >= 256:
            return None
        item += int(coefficient) * multiplier
        multiplier *= 256
    return item


class PrivateExpanderSketchCore(HeavyHitter):
    name = "PrivateExpanderSketch-core"

    def __init__(
        self,
        fallback_rows: int = 5,
        coordinate_constant: float = 1.0,
        hash_range_constant: float = 1.0,
        list_constant: float = 2.0,
        threshold_constant: float = 1.0,
        agreement_fraction: float = 2.0 / 3.0,
        buckets: int = None,
        beam_width: int = 4096,
        candidate_multiplier: int = 12,
        allow_diagnostic_list_recovery: bool = True,
    ):
        self.fallback_rows = int(fallback_rows)
        self.coordinate_constant = float(coordinate_constant)
        self.hash_range_constant = float(hash_range_constant)
        self.list_constant = float(list_constant)
        self.threshold_constant = float(threshold_constant)
        self.agreement_fraction = float(agreement_fraction)
        self.buckets = buckets
        self.beam_width = int(beam_width)
        self.candidate_multiplier = int(candidate_multiplier)
        # Retained for backwards-compatible construction of saved experiment
        # configurations.  The paper branch is now always enabled.
        self.allow_diagnostic_list_recovery = bool(
            allow_diagnostic_list_recovery
        )

    @staticmethod
    def _hash(values, a, b, prime, buckets):
        return ((a * values + b) % prime) % buckets

    @staticmethod
    def _polynomial_hash(values, coefficients, prime, buckets):
        """Evaluate a public k-wise-independent polynomial hash."""
        values = np.asarray(values, dtype=np.int64)
        result = np.zeros(values.size, dtype=np.int64)
        for coefficient in reversed(np.asarray(coefficients, dtype=np.int64)):
            result = (result * values + coefficient) % prime
        return result % buckets

    def _full_domain_branch(
        self, arr, domain_size, epsilon, top_k, seed
    ):
        oracle = Hashtogram(rows=self.fallback_rows)
        domain = np.arange(domain_size, dtype=np.int64)
        estimates = oracle.estimate(
            arr, domain_size, epsilon, seed, domain
        )
        server_started = time.perf_counter()
        order = np.lexsort((domain, -estimates))[:top_k]
        ranking_time = time.perf_counter() - server_started
        parameters = oracle.parameters()
        return {
            "items": (domain[order] + 1).tolist(),
            "estimated_frequencies": estimates[order].tolist(),
            "candidate_count": int(domain_size),
            "runtime_client": float(parameters["runtime_client"]),
            "runtime_server": (
                float(parameters["runtime_server"]) + ranking_time
            ),
            "communication_bits_per_user": 1,
            "diagnostics": {
                "method": self.name,
                "stored_id_base": 1,
                "branch": "paper Section 3.3 n > |X| fallback",
                "condition": {
                    "n": int(arr.size),
                    "domain_size": int(domain_size),
                    "n_greater_than_domain": True,
                },
                "candidate_generation": (
                    "query Hashtogram on every public-domain item and rank "
                    "the resulting private estimates"
                ),
                "privacy": "one epsilon-LDP Hashtogram report per user",
                "hashtogram": parameters,
                "list_recovery_used": False,
                "why_no_list_recovery": (
                    "the paper explicitly selects this branch when n>|X|"
                ),
                "status": (
                    "faithful finite-domain paper branch; -core records that "
                    "no author implementation is available"
                ),
            },
        }

    def _list_recovery_branch(
        self, arr, domain_size, epsilon, top_k, seed
    ):
        log_domain = math.log(max(3, domain_size))
        log_log_domain = max(1.0, math.log(log_domain))
        message_symbols = max(
            1, math.ceil(math.log(max(2, domain_size), 256))
        )
        code_positions = max(
            2 * message_symbols,
            math.ceil(
                self.coordinate_constant
                * log_domain
                / log_log_domain
            ),
        )
        if code_positions >= FIELD:
            raise ValueError("domain representation is too long for GF(257)")
        hash_range = max(
            2,
            1
            << math.ceil(
                math.log2(
                    max(
                        2,
                        math.ceil(
                            log_domain ** self.hash_range_constant
                        ),
                    )
                )
            ),
        )
        bucket_count = int(
            self.buckets
            or max(
                1,
                math.ceil(
                    epsilon
                    * math.sqrt(arr.size)
                    / (log_domain ** 1.5)
                ),
            )
        )
        list_cap = max(1, math.ceil(self.list_constant * log_domain))
        threshold = (
            self.threshold_constant
            * log_log_domain
            * math.sqrt(arr.size)
            / (epsilon * log_domain)
        )
        stream = SeedStream(seed)
        domain_prime = next_prime(domain_size)
        rng = stream.rng("coordinate-hashes")
        hash_a = rng.integers(
            1, domain_prime, size=code_positions, dtype=np.int64
        )
        hash_b = rng.integers(
            0, domain_prime, size=code_positions, dtype=np.int64
        )
        g_independence = max(2, math.ceil(log_domain))
        g_coefficients = stream.rng("outer-g").integers(
            0,
            domain_prime,
            size=g_independence,
            dtype=np.int64,
        )
        outer_buckets = self._polynomial_hash(
            arr, g_coefficients, domain_prime, bucket_count
        )
        code = _rs_symbols(arr, message_symbols, code_positions)

        groups = np.arange(arr.size, dtype=np.int64) % code_positions
        stream.rng("public-groups").shuffle(groups)
        lists = [
            [[] for _ in range(bucket_count)]
            for _ in range(code_positions)
        ]
        client_time = 0.0
        server_time = 0.0
        query_domain = bucket_count * hash_range * FIELD
        recovery_rows = max(5, math.ceil(math.log2(query_domain)))
        category_queries = np.arange(query_domain, dtype=np.int64)
        for position in range(code_positions):
            mask = groups == position
            hashed = self._hash(
                arr[mask],
                hash_a[position],
                hash_b[position],
                domain_prime,
                hash_range,
            )
            categories = (
                (outer_buckets[mask] * hash_range + hashed) * FIELD
                + code[mask, position]
            )
            oracle = Hashtogram(rows=recovery_rows)
            table = oracle.estimate(
                categories,
                query_domain,
                epsilon / 2.0,
                stream.derive(f"recover-p{position}"),
                category_queries,
            ).reshape(bucket_count, hash_range, FIELD)
            client_time += float(oracle.parameters()["runtime_client"])
            server_time += float(oracle.parameters()["runtime_server"])
            best_symbols = np.argmax(table, axis=2)
            best_scores = np.max(table, axis=2)
            for bucket in range(bucket_count):
                entries = [
                    (
                        int(y_value),
                        int(best_symbols[bucket, y_value]),
                        float(best_scores[bucket, y_value]),
                    )
                    for y_value in range(hash_range)
                    if best_scores[bucket, y_value] >= threshold
                ]
                entries.sort(key=lambda entry: (-entry[2], entry[0], entry[1]))
                lists[position][bucket] = entries[:list_cap]

        votes = Counter()
        aggregate_scores = Counter()
        agreement_counts = Counter()
        decode_started = time.perf_counter()
        required_agreement = math.ceil(
            self.agreement_fraction * code_positions
        )
        for bucket in range(bucket_count):
            nonempty_positions = [
                position
                for position in range(code_positions)
                if lists[position][bucket]
            ]
            if len(nonempty_positions) < message_symbols:
                continue
            seed_positions = sorted(
                nonempty_positions,
                key=lambda position: (
                    len(lists[position][bucket]),
                    position,
                ),
            )[:message_symbols]
            points = np.asarray(
                [position + 1 for position in seed_positions],
                dtype=np.int64,
            )
            vandermonde = np.empty(
                (message_symbols, message_symbols), dtype=np.int64
            )
            vandermonde[:, 0] = 1
            for degree in range(1, message_symbols):
                vandermonde[:, degree] = (
                    vandermonde[:, degree - 1] * points
                ) % FIELD
            inverse = _modular_inverse(vandermonde)
            beam = [((), 0.0)]
            for position in seed_positions:
                expanded = [
                    (
                        prefix + (entry[1],),
                        score + entry[2],
                    )
                    for prefix, score in beam
                    for entry in lists[position][bucket]
                ]
                expanded.sort(key=lambda pair: -pair[1])
                beam = expanded[: self.beam_width]
            lookups = [
                {
                    (entry[0], entry[1]): entry[2]
                    for entry in lists[position][bucket]
                }
                for position in range(code_positions)
            ]
            for evaluations, _ in beam:
                coefficients = (
                    inverse.dot(np.asarray(evaluations, dtype=np.int64))
                    % FIELD
                )
                item = _coefficients_to_item(coefficients)
                if item is None or item >= domain_size:
                    continue
                item_array = np.asarray([item], dtype=np.int64)
                mapped_bucket = int(
                    self._polynomial_hash(
                        item_array,
                        g_coefficients,
                        domain_prime,
                        bucket_count,
                    )[0]
                )
                if mapped_bucket != bucket:
                    continue
                symbols = _rs_symbols(
                    item_array, message_symbols, code_positions
                )[0]
                agreement = 0
                score = 0.0
                for position in range(code_positions):
                    y_value = int(
                        self._hash(
                            item_array,
                            hash_a[position],
                            hash_b[position],
                            domain_prime,
                            hash_range,
                        )[0]
                    )
                    entry_score = lookups[position].get(
                        (y_value, int(symbols[position]))
                    )
                    if entry_score is not None:
                        agreement += 1
                        score += entry_score
                if agreement < required_agreement:
                    continue
                votes[item] += 1
                agreement_counts[item] = max(
                    agreement_counts[item], agreement
                )
                aggregate_scores[item] = max(
                    aggregate_scores[item], score
                )
        server_time += time.perf_counter() - decode_started

        max_candidates = max(top_k, self.candidate_multiplier * top_k)
        ranked = sorted(
            votes,
            key=lambda item: (
                -agreement_counts[item],
                -aggregate_scores[item],
                item,
            ),
        )[:max_candidates]
        candidates = np.asarray(ranked, dtype=np.int64)
        verifier = Hashtogram(rows=5)
        if candidates.size:
            estimates = verifier.estimate(
                arr,
                domain_size,
                epsilon / 2.0,
                stream.derive("final-verification"),
                candidates,
            )
            client_time += float(
                verifier.parameters()["runtime_client"]
            )
            server_time += float(
                verifier.parameters()["runtime_server"]
            )
        else:
            estimates = np.empty(0, dtype=float)
        order = np.lexsort((candidates, -estimates))[:top_k]
        return {
            "items": (candidates[order] + 1).tolist(),
            "estimated_frequencies": estimates[order].tolist(),
            "candidate_count": int(candidates.size),
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": 2,
            "diagnostics": {
                "method": self.name,
                "stored_id_base": 1,
                "branch": "paper-structured n <= |X| list recovery",
                "condition": {
                    "n": int(arr.size),
                    "domain_size": int(domain_size),
                    "n_greater_than_domain": False,
                },
                "encoding": (
                    f"Reed-Solomon [{code_positions},{message_symbols},"
                    f"{code_positions - message_symbols + 1}] over GF(257)"
                ),
                "coordinate_count_M": code_positions,
                "hash_range_Y": hash_range,
                "outer_bucket_count_B": bucket_count,
                "list_cap_ell": list_cap,
                "selection_threshold": threshold,
                "hashtogram_rows_per_coordinate": recovery_rows,
                "required_coordinate_agreement": required_agreement,
                "g_independence": g_independence,
                "beam_width": self.beam_width,
                "candidate_recovery": (
                    "per-(m,b,y) private argmax, bounded unique-y lists, "
                    "finite-field interpolation, and all-position agreement"
                ),
                "privacy": (
                    "one epsilon/2 position task plus one epsilon/2 final "
                    "Hashtogram report per user"
                ),
                "list_recovery_used": True,
                "remaining_difference": (
                    "the executable GF(257) Reed-Solomon bounded-list decoder "
                    "instantiates the paper's unique-list-recovery interface; "
                    "the unreleased asymptotic expander/clustering decoder is "
                    "not available as author code"
                ),
                "constant_policy": {
                    "C_M": self.coordinate_constant,
                    "C_Y": self.hash_range_constant,
                    "C_ell": self.list_constant,
                    "C_f": self.threshold_constant,
                    "agreement_fraction": self.agreement_fraction,
                    "source": (
                        "fixed before the five-dataset run; the paper states "
                        "only universal constants and gives no numeric values"
                    ),
                },
                "no_truth_use": (
                    "candidate construction and constants do not access the "
                    "true top-k or true frequencies"
                ),
            },
        }

    def run(self, values, domain_size, epsilon, top_k, seed):
        arr = validate_inputs(values, domain_size, epsilon)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if arr.size > domain_size:
            return self._full_domain_branch(
                arr, domain_size, epsilon, top_k, seed
            )
        return self._list_recovery_branch(
            arr, domain_size, epsilon, top_k, seed
        )
