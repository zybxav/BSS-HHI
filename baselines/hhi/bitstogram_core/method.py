"""Finite-scale implementation of Bitstogram (Bassily et al., Algorithm 8).

The paper only requires a constant-rate binary code correcting a 1/8 fraction
of errors.  We therefore use a public binary code family that scales with the
active domain: the original verified [64,16,17] code for domains up to 2^16,
an exhaustively verified [128,20,38] public linear code for domains up to
2^20, and a deterministic systematic fallback beyond that.  Exact nearest-
codeword decoding is still performed over the active domain.
"""

import math
import time
from typing import Dict

import numpy as np

from baselines.common.error_correcting import (
    CODE_BITS,
    code_configuration,
    decode_nearest,
    encoded_bit,
)
from baselines.common.interfaces import HeavyHitter, validate_inputs
from baselines.common.randomness import SeedStream
from baselines.fo.hashtogram import Hashtogram
from baselines.fo.hadaoracle.oracle import next_power_of_two
from baselines.fo.standard.oracles import _linear_hash_queries


class BitstogramCore(HeavyHitter):
    name = "Bitstogram-core"
    PROFILES = {"finite-topk", "paper-scaled"}

    def __init__(
        self,
        profile: str = "finite-topk",
        failure_probability: float = 0.05,
        rounds: int = None,
        buckets: int = None,
        inner_rows: int = None,
        verification_rows: int = None,
    ):
        self.profile = str(profile)
        self.failure_probability = float(failure_probability)
        self.rounds = None if rounds is None else int(rounds)
        self.buckets = buckets
        self.inner_rows = None if inner_rows is None else int(inner_rows)
        self.verification_rows = (
            None if verification_rows is None else int(verification_rows)
        )

    def _parameters(
        self,
        users: int,
        epsilon: float,
        top_k: int,
        code_bits: int = CODE_BITS,
    ):
        """Resolve either the historical finite profile or paper-scaled one.

        Lemma 6.8 specifies the leading scaling
        ``T = O(sqrt(epsilon*n/(R*log(d'))))`` and ``R = O(log(1/beta))``.
        It leaves constants unspecified.  ``paper-scaled`` therefore uses the
        leading terms with explicit unit constants; it is not presented as a
        literal theorem-constant instantiation.
        """
        if self.profile not in self.PROFILES:
            raise ValueError(
                f"profile must be one of {sorted(self.PROFILES)}"
            )
        if not 0.0 < self.failure_probability < 1.0:
            raise ValueError("failure_probability must be in (0, 1)")
        if self.profile == "finite-topk":
            default_rounds = 3
            default_buckets = max(8, 4 * top_k)
            default_inner_rows = 3
            default_verification_rows = 5
            policy = (
                "finite profile: R=3, T=max(8,4*top_k), inner rows=3, "
                "verification rows=5"
            )
        else:
            default_rounds = max(
                1, int(math.ceil(math.log(1.0 / self.failure_probability)))
            )
            # Algorithm 8 uses a constant-rate code, so its codeword length
            # is the finite counterpart of log(d').  The theorem has a hidden
            # multiplicative constant; this profile records the unit choice.
            default_buckets = max(
                8,
                int(
                    math.ceil(
                        math.sqrt(
                            epsilon
                            * users
                            / (default_rounds * code_bits)
                        )
                    )
                ),
            )
            default_inner_rows = max(
                1, int(math.ceil(math.log(4.0 / self.failure_probability)))
            )
            default_verification_rows = default_inner_rows
            policy = (
                "paper-scaled core: R=ceil(log(1/beta)), "
                "T=ceil(sqrt(epsilon*n/(R*code_bits))) before power-of-two "
                "rounding; Hashtogram rows=ceil(log(4/beta)); all hidden "
                "theorem constants are set explicitly to one"
            )
        rounds = int(self.rounds if self.rounds is not None else default_rounds)
        raw_buckets = int(
            self.buckets if self.buckets is not None else default_buckets
        )
        inner_rows = int(
            self.inner_rows
            if self.inner_rows is not None
            else default_inner_rows
        )
        verification_rows = int(
            self.verification_rows
            if self.verification_rows is not None
            else default_verification_rows
        )
        if rounds < 1 or raw_buckets < 2 or inner_rows < 1:
            raise ValueError("rounds, buckets, and inner_rows must be positive")
        if verification_rows < 1:
            raise ValueError("verification_rows must be positive")
        return {
            "rounds": rounds,
            "raw_buckets": raw_buckets,
            "buckets": next_power_of_two(raw_buckets),
            "inner_rows": inner_rows,
            "verification_rows": verification_rows,
            "policy": policy,
        }

    def run(self, values, domain_size, epsilon, top_k, seed) -> Dict[str, object]:
        arr = validate_inputs(values, domain_size, epsilon)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        code = code_configuration(domain_size)
        params = self._parameters(arr.size, epsilon, top_k, code.code_bits)
        rounds = int(params["rounds"])
        buckets = int(params["buckets"])
        inner_rows = int(params["inner_rows"])
        verification_rows = int(params["verification_rows"])
        stream = SeedStream(seed)
        # Publicly assign the actual domain to the active ECC messages by a
        # bijection.  This removes dependence on numeric label ordering without
        # decoding into inactive 16-bit messages (which would discard genuine
        # candidates after inversion).
        code_rng = stream.rng("public-message-permutation")
        message_for_item = code_rng.permutation(domain_size).astype(np.int64)
        item_for_message = np.empty(domain_size, dtype=np.int64)
        item_for_message[message_for_item] = np.arange(
            domain_size, dtype=np.int64
        )
        encoded_messages = message_for_item[arr]
        hash_rng = stream.rng("outer-hashes")
        outer_bits = int(math.log2(buckets))
        hash_masks = hash_rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=(rounds, outer_bits),
            endpoint=True,
            dtype=np.uint64,
        )
        hash_offsets = hash_rng.integers(
            0, buckets, size=rounds, dtype=np.int64
        )
        hashed_by_round = _linear_hash_queries(
            encoded_messages, hash_masks, hash_offsets
        )

        # Algorithm 8 assigns every user to exactly one (round, code position)
        # discovery task.  All users later make one independent verification
        # report, so epsilon/2 + epsilon/2 gives total epsilon-LDP.
        task_count = rounds * code.code_bits
        group_rng = stream.rng("public-groups")
        groups = np.arange(arr.size, dtype=np.int64) % task_count
        group_rng.shuffle(groups)
        recovered = np.zeros(
            (rounds, buckets, code.code_bits), dtype=np.uint8
        )
        client_time = 0.0
        server_time = 0.0
        group_sizes = []

        for round_id in range(rounds):
            hashed = hashed_by_round[round_id]
            for position in range(code.code_bits):
                task = round_id * code.code_bits + position
                mask = groups == task
                group_sizes.append(int(mask.sum()))
                bits = encoded_bit(encoded_messages[mask], position, config=code)
                categories = hashed[mask] * 2 + bits
                oracle = Hashtogram(rows=inner_rows)
                estimates = oracle.estimate(
                    categories,
                    buckets * 2,
                    epsilon / 2.0,
                    stream.derive(f"discover-r{round_id}-l{position}"),
                ).reshape(buckets, 2)
                recovered[round_id, :, position] = (
                    estimates[:, 1] > estimates[:, 0]
                )
                client_time += float(
                    oracle.parameters()["runtime_client"]
                )
                server_time += float(
                    oracle.parameters()["runtime_server"]
                )

        decode_started = time.perf_counter()
        decoded_messages, distances = decode_nearest(
            recovered.reshape(-1, code.code_bits),
            domain_size,
            config=code,
        )
        server_time += time.perf_counter() - decode_started
        decoded_items = item_for_message[decoded_messages.astype(np.int64)]
        candidates = np.unique(decoded_items)

        verifier = Hashtogram(rows=verification_rows)
        estimates = verifier.estimate(
            arr,
            domain_size,
            epsilon / 2.0,
            stream.derive("independent-final-verification"),
            candidates,
        )
        client_time += float(verifier.parameters()["runtime_client"])
        server_time += float(verifier.parameters()["runtime_server"])
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
                "profile": self.profile,
                "stored_id_base": 1,
                "paper_flow": (
                    "Algorithm 8: public round/position groups, hashed "
                    "(bucket,code-bit) Hashtogram recovery, codeword decode, "
                    "and independent Hashtogram candidate verification"
                ),
                "encoding": {
                    "type": "public binary code family",
                    "family": code.family,
                    "message_bits": code.message_bits,
                    "code_bits": code.code_bits,
                    "rate": code.message_bits / code.code_bits,
                    "minimum_distance": code.minimum_distance,
                    "correctable_adversarial_errors": code.correctable_errors,
                    "paper_requirement": (
                        "correct up to 1/8 of the active public code length"
                    ),
                    "verification": code.verification,
                },
                "decoder": (
                    "exact nearest-codeword search over the active domain "
                    "followed by inversion of a public item/message bijection; "
                    "no truth is used"
                ),
                "public_message_permutation": (
                    "uniform random bijection between item labels and the "
                    "active ECC message domain, derived from the run seed"
                ),
                "rounds": rounds,
                "bucket_count": buckets,
                "raw_bucket_request": params["raw_buckets"],
                "outer_hash_family": (
                    "exact pairwise-independent affine GF(2) map from the "
                        "public ECC message to a power-of-two bucket domain"
                ),
                "code_positions": code.code_bits,
                "discovery_tasks": task_count,
                "users_per_discovery_task_min": min(group_sizes),
                "users_per_discovery_task_max": max(group_sizes),
                "inner_hashtogram_rows": inner_rows,
                "verification_hashtogram_rows": verification_rows,
                "decode_distance_median": float(np.median(distances)),
                "decode_distance_max": int(np.max(distances)),
                "within_unique_radius": (
                    None
                    if code.correctable_errors is None
                    else int(np.sum(distances <= code.correctable_errors))
                ),
                "privacy": (
                    "each user makes one epsilon/2 discovery report and one "
                    "epsilon/2 final-verification report; sequential "
                    "composition is epsilon-LDP"
                ),
                "parameter_policy": params["policy"],
                "remaining_difference": (
                    "the verified finite linear code and exhaustive decoder "
                    "replace an unspecified asymptotically efficient code"
                ),
            },
        }
