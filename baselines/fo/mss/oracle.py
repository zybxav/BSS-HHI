"""End-to-end wrapper over the official AAAI 2026 MSS implementation."""

import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from baselines.common.interfaces import FrequencyOracle, validate_inputs, validate_queries
from baselines.common.randomness import legacy_seed


VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "private-frequency-oracle-rns"


class MSSOracle(FrequencyOracle):
    name = "MSS"

    def __init__(self, channels: Optional[int] = None, max_channels: int = 12):
        self.channels = channels
        self.max_channels = max_channels
        self.last_diagnostics: Dict[str, object] = {}
        self._protocol_cache = {}

    @staticmethod
    def _sample_subset(rng, modulus, residue, omega, include_probability):
        include_true = bool(rng.random() < include_probability)
        other_count = int(omega) - int(include_true)
        other = rng.choice(int(modulus) - 1, size=other_count, replace=False)
        other = other + (other >= int(residue))
        if include_true:
            return np.concatenate(
                (np.asarray([int(residue)], dtype=np.int64), other)
            )
        return other

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
        sys.path.insert(0, str(VENDOR))
        try:
            from ldp_protocols.mss import ModularSubsetSelection
            from scipy.sparse.linalg import lsmr
        finally:
            if sys.path[0] == str(VENDOR):
                sys.path.pop(0)
        with legacy_seed(seed):
            # Moduli are selected from the mechanism seed.  ``run_fo`` reuses
            # one oracle across seeds, so omitting the seed here silently made
            # seeds 102+ reuse seed 101's public modulus design.
            cache_key = (
                domain_size,
                float(epsilon),
                self.channels,
                self.max_channels,
                int(seed),
            )
            protocol = self._protocol_cache.get(cache_key)
            if protocol is None:
                protocol = ModularSubsetSelection(
                    domain_size,
                    epsilon,
                    l=self.channels,
                    l_max=self.max_channels,
                    lambda_reg=None,
                    seed=int(seed),
                    cache_moduli=False,
                )
                self._protocol_cache[cache_key] = protocol
            protocol.seed = int(seed)
            protocol.rng = np.random.default_rng(int(seed))
            started = time.perf_counter()
            channels = protocol.rng.integers(protocol.l, size=arr.size)
            counts = [
                np.zeros(int(modulus), dtype=np.int64)
                for modulus in protocol.moduli
            ]
            channel_sizes = []
            for channel, modulus in enumerate(protocol.moduli):
                modulus = int(modulus)
                local = arr[channels == channel] % modulus
                n_local = int(local.size)
                channel_sizes.append(n_local)
                omega = int(protocol.omega[channel])
                p = float(protocol._p_i[channel])
                for residue in local:
                    residue = int(residue)
                    subset = self._sample_subset(
                        protocol.rng, modulus, residue, omega, p
                    )
                    np.add.at(counts[channel], subset, 1)
            client_time = time.perf_counter() - started
            server_started = time.perf_counter()
            blocks = []
            for channel, modulus in enumerate(protocol.moduli):
                modulus = int(modulus)
                n_local = channel_sizes[channel]
                if n_local:
                    s_hat = (
                        counts[channel] / n_local - protocol._q_i[channel]
                    ) / (
                        protocol._p_i[channel] - protocol._q_i[channel]
                    )
                else:
                    s_hat = np.zeros(modulus, dtype=float)
                blocks.append(s_hat)
            # This is the sufficient-statistic equivalent of the author's
            # _debias_block followed by estimate_raw.  Avoiding report objects
            # changes neither the joint Subset Selection reports nor the
            # author's weighted linear system.
            s_vec = np.concatenate(blocks) * protocol._w_sqrt
            full = lsmr(
                protocol._Aw,
                s_vec,
                damp=math.sqrt(protocol.lambda_reg),
            )[0]
            result = full[queries] * arr.size
            server_time = time.perf_counter() - server_started
        subset_bits = [
            math.ceil(
                (
                    math.lgamma(int(modulus) + 1)
                    - math.lgamma(int(omega) + 1)
                    - math.lgamma(int(modulus) - int(omega) + 1)
                )
                / math.log(2.0)
            )
            for modulus, omega in zip(protocol.moduli, protocol.omega)
        ]
        self.last_diagnostics = {
            "moduli": protocol.moduli.tolist(),
            "channels": protocol.l,
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": math.ceil(math.log2(protocol.l))
            + float(np.mean(subset_bits)),
            "vendor_commit": "824a3f72e360bba45bfb59047f5dab9fd7b2d3cc",
            "estimator": (
                "author estimate_raw weighted sparse LSMR reconstruction"
            ),
            "compatibility": (
                "author analytical modulus selection when channels=None; "
                "exact fixed-cardinality Subset Selection reports aggregated "
                "without retaining report objects"
            ),
            "channel_selection": (
                "author analytical MSE selection"
                if self.channels is None
                else f"explicit fixed-l diagnostic (l={self.channels})"
            ),
            "report_distribution": (
                "exact per-user MSS joint report distribution; no independent-"
                "marginal binomial approximation"
            ),
            "subset_encoding_bits_by_channel": subset_bits,
            "channel_user_counts": channel_sizes,
        }
        return np.asarray(result, dtype=float)

    def parameters(self) -> Dict[str, object]:
        return dict(self.last_diagnostics)
