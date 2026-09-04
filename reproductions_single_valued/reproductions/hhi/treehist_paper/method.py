"""Paper-oriented TreeHist implementation (JMLR 2020, Algorithms 1--3).

Unlike the third-party pure-LDP TreeHistogram variant, this module implements
the paper's signed Count-Sketch hashes, Hadamard row sampling, public
``(level, hash row, Hadamard row)`` assignment, median aggregation, and
independent final frequency reports.
"""

import math
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from reproductions.common.interfaces import HeavyHitter, validate_inputs
from reproductions.fo.hadaoracle.oracle import (
    _uint64_parity,
    hadamard_sign,
    linear_pairwise_hash,
    next_power_of_two,
)


def _node_codes(prefixes: np.ndarray, level: int) -> np.ndarray:
    """Injectively encode ``(level, prefix)`` with a leading sentinel bit."""
    return (np.int64(1) << np.int64(level)) + np.asarray(
        prefixes, dtype=np.int64
    )


def _fwht_rows(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if out.ndim != 2 or out.shape[1] == 0:
        raise ValueError("expected a nonempty two-dimensional table")
    if out.shape[1] & (out.shape[1] - 1):
        raise ValueError("Hadamard width must be a power of two")
    step = 1
    while step < out.shape[1]:
        for start in range(0, out.shape[1], 2 * step):
            left = out[:, start : start + step].copy()
            right = out[:, start + step : start + 2 * step].copy()
            out[:, start : start + step] = left + right
            out[:, start + step : start + 2 * step] = left - right
        step *= 2
    return out


class TreeHistPaper(HeavyHitter):
    """TreeHist core mechanism with explicit finite-sample profiles.

    ``topk-adapted`` retains the paper mechanism but replaces theorem-threshold
    pruning by a data-independent ``ceil(sqrt(n))`` candidate bound.  This is
    the default for an operational F1@k experiment.

    ``paper-experimental`` uses the paper's documented finite experiment
    settings (285 hashes, width near sqrt(n), pruning threshold 15*sqrt(n))
    while retaining Algorithm 1's Hadamard compression.

    ``theorem`` uses the literal constants from Algorithms 1--3.  At current
    100k-user scales those constants are normally vacuous and the profile is
    intended for protocol diagnostics rather than F1 comparison.
    """

    name = "TreeHist-paper"
    PROFILES = {"topk-adapted", "paper-experimental", "theorem"}

    def __init__(
        self,
        profile: str = "topk-adapted",
        failure_probability: float = 0.05,
        hash_count: Optional[int] = None,
        sketch_width: Optional[int] = None,
        candidate_limit: Optional[int] = None,
        experimental_threshold_multiplier: float = 15.0,
    ):
        self.profile = str(profile)
        self.failure_probability = float(failure_probability)
        self.hash_count = None if hash_count is None else int(hash_count)
        self.sketch_width = (
            None if sketch_width is None else int(sketch_width)
        )
        self.candidate_limit = (
            None if candidate_limit is None else int(candidate_limit)
        )
        self.experimental_threshold_multiplier = float(
            experimental_threshold_multiplier
        )
        self.name = {
            "topk-adapted": "TreeHist-paper (top-k adaptation)",
            "paper-experimental": "TreeHist-paper (experimental profile)",
            "theorem": "TreeHist-paper (theorem profile)",
        }.get(self.profile, "TreeHist-paper")

    def _parameters(
        self, users: int, domain_size: int, epsilon: float
    ) -> Dict[str, object]:
        if self.profile not in self.PROFILES:
            raise ValueError(
                f"profile must be one of {sorted(self.PROFILES)}"
            )
        if not 0.0 < self.failure_probability < 1.0:
            raise ValueError("failure_probability must be in (0, 1)")
        log_term = math.log(users / self.failure_probability)
        levels = max(1, int(math.ceil(math.log2(domain_size))))
        if self.profile == "theorem":
            hashes = int(math.ceil(110.0 * log_term))
            raw_width = int(
                math.ceil(48.0 * math.sqrt(users / log_term))
            )
            eta = (
                147.0
                / epsilon
                * math.sqrt(
                    users
                    * log_term
                    * math.log(max(2, domain_size))
                )
            )
            pruning_threshold = 2.0 * eta
            selection = "paper threshold: estimate >= 2*eta"
        elif self.profile == "paper-experimental":
            hashes = 285
            raw_width = int(math.ceil(math.sqrt(users)))
            eta = None
            pruning_threshold = (
                self.experimental_threshold_multiplier
                * math.sqrt(users)
            )
            selection = (
                "paper finite-experiment threshold: "
                f"{self.experimental_threshold_multiplier}*sqrt(n)"
            )
        else:
            hashes = 285
            raw_width = int(math.ceil(math.sqrt(users)))
            eta = None
            pruning_threshold = None
            selection = (
                "top-k metric adaptation: retain the highest estimated "
                "ceil(sqrt(n)) prefixes at each level"
            )
        if self.hash_count is not None:
            hashes = self.hash_count
        if self.sketch_width is not None:
            raw_width = self.sketch_width
        if hashes < 1 or raw_width < 2:
            raise ValueError("hash_count and sketch_width must be positive")
        width = next_power_of_two(raw_width)
        limit = int(
            self.candidate_limit
            if self.candidate_limit is not None
            else math.ceil(math.sqrt(users))
        )
        if limit < 1:
            raise ValueError("candidate_limit must be positive")
        return {
            "levels": levels,
            "hashes": hashes,
            "raw_width": raw_width,
            "width": width,
            "eta": eta,
            "pruning_threshold": pruning_threshold,
            "candidate_limit": limit,
            "selection": selection,
            "theorem_precondition_n_ge_48_t_L": (
                users >= 48 * hashes * levels
            ),
        }

    @staticmethod
    def _hash_and_sign(
        nodes: np.ndarray,
        masks: np.ndarray,
        offsets: np.ndarray,
        sign_mask: np.uint64,
        sign_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        columns = linear_pairwise_hash(nodes, masks, offsets)
        sign_bits = _uint64_parity(
            np.asarray(nodes, dtype=np.uint64) & sign_mask
        )
        sign_bits ^= int(sign_offset)
        signs = 1.0 - 2.0 * sign_bits
        return columns, signs

    def _make_reports(
        self,
        arr: np.ndarray,
        levels: np.ndarray,
        hash_rows: np.ndarray,
        hadamard_rows: np.ndarray,
        level_count: int,
        hash_masks: np.ndarray,
        hash_offsets: np.ndarray,
        sign_masks: np.ndarray,
        sign_offsets: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prefix_values = arr >> (level_count - levels)
        prefix_nodes = (
            np.left_shift(np.int64(1), levels) + prefix_values
        )
        leaf_nodes = _node_codes(arr, level_count)
        prune_signal = np.empty(arr.size, dtype=float)
        final_signal = np.empty(arr.size, dtype=float)
        for row in range(hash_masks.shape[0]):
            selected = hash_rows == row
            if not np.any(selected):
                continue
            prune_columns, prune_signs = self._hash_and_sign(
                prefix_nodes[selected],
                hash_masks[row],
                hash_offsets[row],
                sign_masks[row],
                sign_offsets[row],
            )
            final_columns, final_signs = self._hash_and_sign(
                leaf_nodes[selected],
                hash_masks[row],
                hash_offsets[row],
                sign_masks[row],
                sign_offsets[row],
            )
            public_rows = hadamard_rows[selected]
            prune_signal[selected] = prune_signs * hadamard_sign(
                public_rows, prune_columns
            )
            final_signal[selected] = final_signs * hadamard_sign(
                public_rows, final_columns
            )
        keep_probability = math.exp(epsilon / 2.0) / (
            math.exp(epsilon / 2.0) + 1.0
        )
        prune_reports = np.where(
            rng.random(arr.size) < keep_probability,
            prune_signal,
            -prune_signal,
        ).astype(np.int8)
        final_reports = np.where(
            rng.random(arr.size) < keep_probability,
            final_signal,
            -final_signal,
        ).astype(np.int8)
        return prune_reports, final_reports

    @staticmethod
    def _aggregate_tables(
        reports: np.ndarray,
        hash_rows: np.ndarray,
        hadamard_rows: np.ndarray,
        hashes: int,
        width: int,
        selected: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if selected is None:
            selected = np.ones(reports.size, dtype=bool)
        flat = hash_rows[selected] * width + hadamard_rows[selected]
        compressed = np.bincount(
            flat,
            weights=reports[selected],
            minlength=hashes * width,
        ).reshape(hashes, width)
        return _fwht_rows(compressed)

    def _query(
        self,
        prefixes: np.ndarray,
        level: int,
        tables: np.ndarray,
        gamma: float,
        a_epsilon: float,
        hash_masks: np.ndarray,
        hash_offsets: np.ndarray,
        sign_masks: np.ndarray,
        sign_offsets: np.ndarray,
    ) -> np.ndarray:
        nodes = _node_codes(prefixes, level)
        row_estimates = np.empty(
            (hash_masks.shape[0], prefixes.size), dtype=float
        )
        for row in range(hash_masks.shape[0]):
            columns, signs = self._hash_and_sign(
                nodes,
                hash_masks[row],
                hash_offsets[row],
                sign_masks[row],
                sign_offsets[row],
            )
            row_estimates[row] = (
                gamma
                * a_epsilon
                * signs
                * tables[row, columns]
            )
        return np.median(row_estimates, axis=0)

    def run(
        self,
        values: Iterable[int],
        domain_size: int,
        epsilon: float,
        top_k: int,
        seed: int,
    ) -> Dict[str, object]:
        arr = validate_inputs(values, domain_size, epsilon)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        params = self._parameters(arr.size, domain_size, epsilon)
        level_count = int(params["levels"])
        if level_count >= 62:
            raise ValueError("TreeHist node encoding supports fewer than 62 levels")
        hashes = int(params["hashes"])
        width = int(params["width"])
        output_bits = int(math.log2(width))
        rng = np.random.default_rng(int(seed))

        client_started = time.perf_counter()
        levels = rng.integers(
            1, level_count + 1, size=arr.size, dtype=np.int64
        )
        hash_rows = rng.integers(
            0, hashes, size=arr.size, dtype=np.int64
        )
        hadamard_rows = rng.integers(
            0, width, size=arr.size, dtype=np.int64
        )
        hash_masks = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=(hashes, output_bits),
            endpoint=True,
            dtype=np.uint64,
        )
        hash_offsets = rng.integers(
            0, 2, size=(hashes, output_bits), dtype=np.int64
        )
        sign_masks = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=hashes,
            endpoint=True,
            dtype=np.uint64,
        )
        sign_offsets = rng.integers(
            0, 2, size=hashes, dtype=np.int64
        )
        prune_reports, final_reports = self._make_reports(
            arr,
            levels,
            hash_rows,
            hadamard_rows,
            level_count,
            hash_masks,
            hash_offsets,
            sign_masks,
            sign_offsets,
            epsilon,
            rng,
        )
        client_time = time.perf_counter() - client_started

        server_started = time.perf_counter()
        a_epsilon = (
            math.exp(epsilon / 2.0) + 1.0
        ) / (
            math.exp(epsilon / 2.0) - 1.0
        )
        prefixes = np.asarray([0], dtype=np.int64)
        layer_diagnostics = []
        for level in range(1, level_count + 1):
            expanded = np.concatenate((2 * prefixes, 2 * prefixes + 1))
            subtree_shift = level_count - level
            expanded = np.unique(
                expanded[
                    np.left_shift(expanded, subtree_shift) < domain_size
                ]
            )
            if expanded.size == 0:
                prefixes = expanded
                break
            selected = levels == level
            tables = self._aggregate_tables(
                prune_reports,
                hash_rows,
                hadamard_rows,
                hashes,
                width,
                selected,
            )
            estimates = self._query(
                expanded,
                level,
                tables,
                hashes * level_count,
                a_epsilon,
                hash_masks,
                hash_offsets,
                sign_masks,
                sign_offsets,
            )
            if self.profile == "topk-adapted":
                order = np.lexsort((expanded, -estimates))
                keep = order[
                    : min(int(params["candidate_limit"]), order.size)
                ]
            else:
                keep = np.flatnonzero(
                    estimates >= float(params["pruning_threshold"])
                )
            prefixes = expanded[keep]
            layer_diagnostics.append(
                {
                    "level": level,
                    "users": int(selected.sum()),
                    "queried_prefixes": int(expanded.size),
                    "surviving_prefixes": int(prefixes.size),
                }
            )
            if prefixes.size == 0:
                break

        candidates = prefixes[prefixes < domain_size]
        if candidates.size:
            final_tables = self._aggregate_tables(
                final_reports,
                hash_rows,
                hadamard_rows,
                hashes,
                width,
            )
            final_estimates = self._query(
                candidates,
                level_count,
                final_tables,
                hashes,
                a_epsilon,
                hash_masks,
                hash_offsets,
                sign_masks,
                sign_offsets,
            )
            order = np.lexsort((candidates, -final_estimates))[:top_k]
            items = candidates[order]
            estimates = final_estimates[order]
        else:
            items = np.asarray([], dtype=np.int64)
            estimates = np.asarray([], dtype=float)
        server_time = time.perf_counter() - server_started

        return {
            "items": (items + 1).tolist(),
            "estimated_frequencies": estimates.tolist(),
            "candidate_count": int(candidates.size),
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": 2,
            "diagnostics": {
                "method": self.name,
                "profile": self.profile,
                "paper_algorithms": "JMLR Algorithms 1, 2, and 3",
                "stored_id_base": 1,
                "internal_id_base": 0,
                "levels": level_count,
                "hash_count_t": hashes,
                "hadamard_width_m": width,
                "raw_requested_width": params["raw_width"],
                "hash_family": (
                    "pairwise-independent affine GF(2) hashes over injective "
                    "(prefix length, prefix bits) node codes"
                ),
                "selection": params["selection"],
                "eta": params["eta"],
                "pruning_threshold": params["pruning_threshold"],
                "candidate_limit": (
                    params["candidate_limit"]
                    if self.profile == "topk-adapted"
                    else None
                ),
                "a_epsilon": a_epsilon,
                "theorem_precondition_n_ge_48_t_L": (
                    params["theorem_precondition_n_ge_48_t_L"]
                ),
                "privacy": (
                    "two independent epsilon/2 randomized-response bits per "
                    "user (pruning and final estimation); total epsilon-LDP"
                ),
                "public_assignment": (
                    "each user samples one shared (level, hash row, "
                    "Hadamard row) triple reused across the two phases"
                ),
                "aggregation": (
                    "gamma=tL for pruning and gamma=t for final estimation; "
                    "signed Count-Sketch/Hadamard estimates use median over t"
                ),
                "metric_adapter": (
                    "the returned succinct histogram is ranked by the "
                    "independent final estimates and truncated to top_k"
                ),
                "paper_experiment_difference": (
                    "the paper's Section 4 experiments omitted Hadamard "
                    "compression; this implementation retains Algorithm 1"
                ),
                "layers": layer_diagnostics,
            },
        }
