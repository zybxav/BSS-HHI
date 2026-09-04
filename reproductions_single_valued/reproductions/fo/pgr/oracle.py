"""Wrapper over the MSS repository's BSD-licensed Python PGR translation."""

import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from reproductions.common.interfaces import FrequencyOracle, validate_inputs, validate_queries
from reproductions.common.randomness import legacy_seed


VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "private-frequency-oracle-rns"


class PGROracle(FrequencyOracle):
    name = "PGR (MSS Python port)"

    def __init__(self):
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
        sys.path.insert(0, str(VENDOR))
        try:
            from ldp_protocols.pgr import ProjectiveGeometryResponse
        finally:
            if sys.path[0] == str(VENDOR):
                sys.path.pop(0)
        with legacy_seed(seed):
            protocol = ProjectiveGeometryResponse(domain_size, epsilon, seed=int(seed))
            started = time.perf_counter()
            reports = protocol.obfuscate(arr)
            client_time = time.perf_counter() - started
            server_started = time.perf_counter()
            full = protocol.estimate_raw(reports) * arr.size
            result = full[queries]
            server_time = time.perf_counter() - server_started
        self.last_diagnostics = {
            "q": protocol.q,
            "t": protocol.t,
            "expanded_domain": protocol.k,
            "runtime_client": client_time,
            "runtime_server": server_time,
            "communication_bits_per_user": math.ceil(math.log2(protocol.k)),
            "python_port_commit": "824a3f72e360bba45bfb59047f5dab9fd7b2d3cc",
            "upstream_cpp_commit": "f043e0c3ba180e05ce0883e93e0863c699599e17",
            "implementation_scope": (
                "MSS authors' Python port of the official PGR C++ algorithm, "
                "with local correctness fixes matching the official C++ "
                "prime selection and uniform sampling of nonzero projective "
                "representatives"
            ),
            "report_sampler": (
                "uniform over the preferred/non-preferred projective sets; "
                "the all-zero vector is rejected after enforcing the requested "
                "dot product"
            ),
        }
        return np.asarray(result, dtype=float)

    def parameters(self) -> Dict[str, object]:
        return dict(self.last_diagnostics)
