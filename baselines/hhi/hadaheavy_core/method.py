"""Large-radix layered prefix search backed by HadaOracle."""

import math
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from baselines.common.interfaces import HeavyHitter, validate_inputs
from baselines.common.randomness import SeedStream
from baselines.fo.hadaoracle import HadaOracle


def radix_digits(values: np.ndarray, base: int, width: int) -> np.ndarray:
    result = np.empty((values.size, width), dtype=np.int64)
    work = values.copy()
    for position in range(width - 1, -1, -1):
        result[:, position] = work % base
        work //= base
    return result


def prefix_ids(digits: np.ndarray, base: int, length: int) -> np.ndarray:
    out = np.zeros(digits.shape[0], dtype=np.int64)
    for position in range(length):
        out = out * base + digits[:, position]
    return out


class HadaHeavyCore(HeavyHitter):
    name = "HadaHeavy-core"

    def __init__(
        self,
        base: int = None,
        failure_probability: float = 0.05,
        threshold_constant: float = 1.0,
        oracle_constant_ck: float = 8.0,
    ):
        self.base = base
        self.failure_probability = float(failure_probability)
        self.threshold_constant = float(threshold_constant)
        self.oracle_constant_ck = float(oracle_constant_ck)

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
        if not 0.0 < self.failure_probability < 1.0:
            raise ValueError("failure_probability must be in (0, 1)")
        if self.threshold_constant <= 0 or self.oracle_constant_ck <= 0:
            raise ValueError("paper constants must be positive")
        base = int(self.base or max(2, round(math.sqrt(arr.size))))
        width = max(1, math.ceil(math.log(max(2, domain_size), base)))
        beta0 = self.failure_probability / (arr.size * width)
        lambda0 = (
            self.threshold_constant
            / epsilon
            * math.sqrt(
                arr.size
                * math.log(max(2, domain_size))
                * (
                    math.log(arr.size / self.failure_probability)
                    / math.log(max(2, arr.size))
                )
            )
        )
        pruning_threshold = 2.0 * lambda0
        digits = radix_digits(arr, base, width)
        stream = SeedStream(seed)
        rng = stream.rng("user-groups")
        # Wu-Wirth partitions users into exactly L layer groups.  The last
        # layer already estimates complete item frequencies; no extra
        # verification group is present in Algorithm HadaHeavy.
        groups = np.arange(arr.size, dtype=np.int64) % width
        rng.shuffle(groups)
        candidates = np.array([0], dtype=np.int64)
        layer_diagnostics: List[Dict[str, object]] = []
        client_time = 0.0
        server_time = 0.0
        final_layer_estimates = np.array([], dtype=float)
        final_layer_users = 0

        for layer in range(1, width + 1):
            mask = groups == layer - 1
            observed = prefix_ids(digits[mask], base, layer)
            expanded = (
                np.arange(base, dtype=np.int64)
                if layer == 1
                else (candidates[:, None] * base + np.arange(base)[None, :]).ravel()
            )
            limit = min(base**layer, math.ceil(domain_size / (base ** (width - layer))))
            expanded = np.unique(expanded[expanded < limit])
            if observed.size == 0 or expanded.size == 0:
                candidates = np.array([], dtype=np.int64)
                break
            oracle = HadaOracle(
                failure_probability=beta0,
                constant_ck=self.oracle_constant_ck,
            )
            estimates = oracle.estimate(
                observed,
                domain_size=base**layer,
                epsilon=epsilon,
                seed=stream.derive(f"layer-{layer}"),
                query_items=expanded,
            )
            scaled_estimates = estimates * width
            keep = np.flatnonzero(scaled_estimates >= pruning_threshold)
            candidates = expanded[keep]
            if layer == width:
                final_layer_estimates = scaled_estimates[keep]
                final_layer_users = int(mask.sum())
            client_time += float(oracle.parameters()["runtime_client"])
            server_time += float(oracle.parameters()["runtime_server"])
            layer_diagnostics.append(
                {
                    "layer": layer,
                    "users": int(mask.sum()),
                    "expanded_count": int(expanded.size),
                    "candidate_prefix_count": int(candidates.size),
                    "pruning_threshold": pruning_threshold,
                }
            )

        candidates = candidates[candidates < domain_size]
        final_estimates = final_layer_estimates
        order = np.lexsort((candidates, -final_estimates))[:top_k]
        items_zero = candidates[order]
        estimates = final_estimates[order]
        return {
            "items": (items_zero + 1).tolist(),
            "estimated_frequencies": estimates.tolist(),
            "candidate_count": int(candidates.size),
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": 1,
            "diagnostics": {
                "method": self.name,
                "stored_id_base": 1,
                "internal_id_base": 0,
                "base": base,
                "digits": width,
                "selection": (
                    "paper threshold rule: retain prefixes with estimated "
                    "frequency at least 2*lambda_0"
                ),
                "failure_probability_beta": self.failure_probability,
                "oracle_failure_probability_beta0": beta0,
                "threshold_constant_C_lambda": self.threshold_constant,
                "lambda_0": lambda0,
                "pruning_threshold_2_lambda_0": pruning_threshold,
                "oracle_constant_CK": self.oracle_constant_ck,
                "privacy": (
                    "Wu-Wirth disjoint L layer groups; each user sends one "
                    "epsilon-LDP report"
                ),
                "layers": layer_diagnostics,
                "verification": (
                    "complete-item estimates from the final search layer, "
                    "matching Algorithm HadaHeavy"
                ),
                "verification_users": final_layer_users,
                "final_candidate_count": int(candidates.size),
                "remaining_difference": (
                    "the proof explicitly instantiates C_K=8; C_lambda is "
                    "not numerically specified in the paper and remains an "
                    "explicit finite-scale parameter rather than a top-k beam"
                ),
            },
        }
