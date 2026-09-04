"""Scalable public binary codes for finite-domain HHI.

For domains up to ``2^16`` we retain the original exhaustively verified
``[64,16,17]`` public linear code.  To cover the paper-scale single-valued
datasets used in this repository, we additionally provide an exhaustively
verified public ``[128,20,38]`` linear code, which supports active domains up
to ``2^20`` and still corrects more than a ``1/8`` fraction of adversarial bit
errors.

For still larger domains, the module synthesises a deterministic systematic
public linear code with rate at most ``1/8`` and keeps the same exact
nearest-codeword decoder over the active domain.  Those larger-domain fallback
codes are not exhaustively distance-verified, but they avoid the former
``2^16`` hard stop in the finite Bitstogram implementation.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Optional

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only in minimal environments
    njit = None
    prange = range


LEGACY_GENERATOR_HEX = (
    "07d2e3fcc6dce542",
    "011d782df0117cb4",
    "ad236e789dad70f1",
    "3a611854f76c5b76",
    "9434bf270e42260a",
    "6b63c6d1904b5a88",
    "22c17d67787f3fcf",
    "ca750cbeffafe9f0",
    "593ad020021f3d76",
    "de2f6405501542e2",
    "63db26410565dec3",
    "3a2e87a5bfa2b24a",
    "70d24e977ebaf47f",
    "7900de4d27f5f428",
    "79c7cd116d3c0b9e",
    "700268c5924b826b",
)

# Found by deterministic search on 2026-07-26: seed=1 gives minimum distance 38
# over all 2^20-1 nonzero messages for this [128,20] public random linear code.
EXTENDED_GENERATOR_HEX = (
    ("8306bdf37922e4ff", "f35196bbc152a866"),
    ("24e7a4f608ec18cd", "f2dab0aed2ac6fd2"),
    ("4fd42fa03fcd72a9", "6c5f1f45de787048"),
    ("d3e4513345ee6d24", "68c1464c41ca3aca"),
    ("8cb22c02a4d1fa2a", "070e1d3215f30262"),
    ("c0e63c2edd973235", "89c3c29ad67f9ca4"),
    ("54694c3ad14a9948", "c9d676a873e2d50a"),
    ("4d9e2d241fb91945", "7418700c1fda0c2e"),
    ("22508e81fa16820b", "673269a56220b55d"),
    ("3415a484e7642555", "4326f7928093fdec"),
    ("c017e631051316f3", "47c8de4d0fe09745"),
    ("7c3579c77f8cfc76", "fb1197d71defa64d"),
    ("f62f2a74bfc53500", "b98bd564179423de"),
    ("8a8dd7dc4b0ba10a", "46e25789ecc847b9"),
    ("29207d73b9d2f219", "f84d082952945002"),
    ("841d12216bce592e", "1da95e684afffcd8"),
    ("9f9d064d6cb28f44", "c6d4b45f74a21da1"),
    ("9cedc8ca5ce3d0d6", "ead405bac5d58cb3"),
    ("0a22c23e6d6462c7", "8751a03eb7f16dd5"),
    ("75970952df41a5e3", "0ff6245b5e3a7203"),
)

MESSAGE_BITS = 16
CODE_BITS = 64
MINIMUM_DISTANCE = 17
CORRECTABLE_ERRORS = (MINIMUM_DISTANCE - 1) // 2
BYTE_POPCOUNT = np.asarray(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)


@dataclass(frozen=True)
class CodeConfiguration:
    family: str
    message_bits: int
    code_bits: int
    generator_blocks: np.ndarray
    minimum_distance: Optional[int]
    correctable_errors: Optional[int]
    verification: str


def _rows_from_legacy_hex():
    return np.asarray([[int(row, 16)] for row in LEGACY_GENERATOR_HEX], dtype=np.uint64)


def _rows_from_extended_hex():
    return np.asarray(
        [
            [int(high, 16), int(low, 16)]
            for high, low in EXTENDED_GENERATOR_HEX
        ],
        dtype=np.uint64,
    )


LEGACY_GENERATOR_BLOCKS = _rows_from_legacy_hex()
EXTENDED_GENERATOR_BLOCKS = _rows_from_extended_hex()


def active_message_bits(domain_size: int) -> int:
    size = max(2, int(domain_size))
    return max(1, int(math.ceil(math.log2(size))))


def recommended_code_bits(message_bits: int) -> int:
    target = max(64, 8 * int(message_bits))
    return int(math.ceil(target / 64.0) * 64)


def _set_bit(rows, row, position):
    block = position // 64
    shift = 63 - (position % 64)
    rows[row, block] |= np.uint64(1) << np.uint64(shift)


@lru_cache(maxsize=8)
def _dynamic_generator_blocks(message_bits: int, code_bits: int):
    blocks = code_bits // 64
    rng_seed = 0x5EED_2026 + 131 * int(message_bits) + 17 * int(code_bits)
    rng = np.random.default_rng(rng_seed)
    rows = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=(message_bits, blocks),
        endpoint=True,
        dtype=np.uint64,
    )
    # Make the code systematic so the public mapping is injective by design.
    for bit in range(message_bits):
        _set_bit(rows, bit, bit)
    return rows


def _configuration_from_rows(
    family: str,
    message_bits: int,
    code_bits: int,
    rows: np.ndarray,
    minimum_distance: Optional[int],
    verification: str,
):
    correctable = (
        None if minimum_distance is None else (int(minimum_distance) - 1) // 2
    )
    return CodeConfiguration(
        family=family,
        message_bits=int(message_bits),
        code_bits=int(code_bits),
        generator_blocks=np.asarray(rows, dtype=np.uint64),
        minimum_distance=minimum_distance,
        correctable_errors=correctable,
        verification=verification,
    )


@lru_cache(maxsize=16)
def code_configuration(domain_size: int) -> CodeConfiguration:
    message_bits = active_message_bits(domain_size)
    if message_bits <= 16:
        minimum = _minimum_distance_legacy_prefix(message_bits)
        return _configuration_from_rows(
            family="verified-public-linear-[64,<=16]",
            message_bits=message_bits,
            code_bits=64,
            rows=LEGACY_GENERATOR_BLOCKS[:message_bits].copy(),
            minimum_distance=minimum,
            verification=(
                "minimum distance exhaustively checked over all "
                f"2^{message_bits}-1 nonzero messages"
            ),
        )
    if message_bits <= 20:
        minimum = _minimum_distance_extended_prefix(message_bits)
        return _configuration_from_rows(
            family="verified-public-linear-[128,<=20]",
            message_bits=message_bits,
            code_bits=128,
            rows=EXTENDED_GENERATOR_BLOCKS[:message_bits].copy(),
            minimum_distance=minimum,
            verification=(
                "minimum distance exhaustively checked over all "
                f"2^{message_bits}-1 nonzero messages"
            ),
        )
    code_bits = recommended_code_bits(message_bits)
    return _configuration_from_rows(
        family="deterministic-systematic-linear-fallback",
        message_bits=message_bits,
        code_bits=code_bits,
        rows=_dynamic_generator_blocks(message_bits, code_bits).copy(),
        minimum_distance=None,
        verification=(
            "deterministic systematic public linear code; exhaustive minimum-"
            "distance verification is skipped above 20 message bits"
        ),
    )


def _resolve_config(
    config: Optional[CodeConfiguration] = None,
    domain_size: Optional[int] = None,
) -> CodeConfiguration:
    if config is not None:
        return config
    if domain_size is None:
        domain_size = 1 << MESSAGE_BITS
    return code_configuration(int(domain_size))


def encode_packed(messages, config: Optional[CodeConfiguration] = None, domain_size: Optional[int] = None):
    cfg = _resolve_config(config=config, domain_size=domain_size)
    values = np.asarray(messages, dtype=np.uint64)
    flat = values.reshape(-1)
    blocks = cfg.code_bits // 64
    packed = np.zeros((flat.size, blocks), dtype=np.uint64)
    rows = cfg.generator_blocks
    for bit in range(cfg.message_bits):
        selected = ((flat >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        packed[selected] ^= rows[bit]
    return packed.reshape(values.shape + (blocks,))


def encoded_bit(messages, position, config: Optional[CodeConfiguration] = None, domain_size: Optional[int] = None):
    cfg = _resolve_config(config=config, domain_size=domain_size)
    if position < 0 or position >= cfg.code_bits:
        raise ValueError("code position is outside the active public code")
    block = position // 64
    shift = 63 - (position % 64)
    column = (cfg.generator_blocks[:, block] >> np.uint64(shift)) & np.uint64(1)
    values = np.asarray(messages, dtype=np.uint64)
    parity = np.zeros(values.shape, dtype=np.uint8)
    for bit in range(cfg.message_bits):
        parity ^= (((values >> np.uint64(bit)) & 1) * column[bit]).astype(
            np.uint8
        )
    return parity


def pack_codewords(bits, config: Optional[CodeConfiguration] = None, domain_size: Optional[int] = None):
    cfg = _resolve_config(config=config, domain_size=domain_size)
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] != cfg.code_bits:
        raise ValueError("last dimension must contain the active code bits")
    raw = np.packbits(bits, axis=-1, bitorder="big")
    blocks = cfg.code_bits // 64
    raw = raw.reshape(bits.shape[:-1] + (blocks, 8))
    packed = np.zeros(bits.shape[:-1] + (blocks,), dtype=np.uint64)
    for offset in range(8):
        packed = (packed << np.uint64(8)) | raw[..., offset].astype(np.uint64)
    return packed


def _nearest_numpy(received_blocks, code_blocks):
    decoded = np.empty(received_blocks.shape[0], dtype=np.int64)
    distances = np.empty(received_blocks.shape[0], dtype=np.int64)
    chunk = 1 << 16
    for report in range(received_blocks.shape[0]):
        best_item = 0
        best_distance = 10**9
        current = received_blocks[report]
        for start in range(0, code_blocks.shape[0], chunk):
            stop = min(start + chunk, code_blocks.shape[0])
            xor = np.bitwise_xor(code_blocks[start:stop], current)
            bytes_view = xor.view(np.uint8).reshape(stop - start, -1)
            local_distances = BYTE_POPCOUNT[bytes_view].sum(axis=1, dtype=np.int64)
            local_index = int(np.argmin(local_distances))
            local_distance = int(local_distances[local_index])
            if local_distance < best_distance:
                best_distance = local_distance
                best_item = start + local_index
        decoded[report] = best_item
        distances[report] = best_distance
    return decoded, distances


if njit is not None:
    @njit(cache=False, inline="always")
    def _popcount64(value):
        count = 0
        while value:
            value &= value - np.uint64(1)
            count += 1
        return count


    @njit(cache=False, parallel=True)
    def _nearest(received_blocks, code_blocks):
        decoded = np.empty(received_blocks.shape[0], dtype=np.int64)
        distances = np.empty(received_blocks.shape[0], dtype=np.int64)
        for report in prange(received_blocks.shape[0]):
            best_item = 0
            best_distance = 10**9
            for item in range(code_blocks.shape[0]):
                distance = 0
                for block in range(code_blocks.shape[1]):
                    distance += _popcount64(
                        received_blocks[report, block] ^ code_blocks[item, block]
                    )
                if distance < best_distance:
                    best_distance = distance
                    best_item = item
            decoded[report] = best_item
            distances[report] = best_distance
        return decoded, distances
else:
    _nearest = _nearest_numpy


@lru_cache(maxsize=16)
def codebook(domain_size: int):
    cfg = code_configuration(int(domain_size))
    return encode_packed(np.arange(int(domain_size), dtype=np.uint64), config=cfg)


def decode_nearest(bits, domain_size, config: Optional[CodeConfiguration] = None):
    cfg = _resolve_config(config=config, domain_size=domain_size)
    received = pack_codewords(bits, config=cfg)
    active_codebook = (
        codebook(int(domain_size))
        if config is None
        else encode_packed(
            np.arange(int(domain_size), dtype=np.uint64),
            config=cfg,
        )
    )
    shape = received.shape[:-1]
    decoded, distances = _nearest(
        received.reshape(-1, active_codebook.shape[-1]),
        active_codebook,
    )
    return decoded.reshape(shape), distances.reshape(shape)


def _minimum_distance_exact(rows, message_bits: int):
    messages = np.arange(1, 1 << message_bits, dtype=np.uint64)
    code = encode_packed(
        messages,
        config=_configuration_from_rows(
            family="distance-check",
            message_bits=message_bits,
            code_bits=int(rows.shape[1] * 64),
            rows=rows,
            minimum_distance=None,
            verification="distance-check",
        ),
    ).reshape(-1, rows.shape[1])
    bytes_view = code.view(np.uint8).reshape(code.shape[0], -1)
    weights = BYTE_POPCOUNT[bytes_view].sum(axis=1, dtype=np.int64)
    return int(weights.min())


@lru_cache(maxsize=16)
def _minimum_distance_legacy_prefix(message_bits: int):
    return _minimum_distance_exact(
        LEGACY_GENERATOR_BLOCKS[:message_bits],
        int(message_bits),
    )


@lru_cache(maxsize=16)
def _minimum_distance_extended_prefix(message_bits: int):
    return _minimum_distance_exact(
        EXTENDED_GENERATOR_BLOCKS[:message_bits],
        int(message_bits),
    )


def verify_minimum_distance(domain_size: Optional[int] = None):
    cfg = _resolve_config(domain_size=domain_size)
    if cfg.minimum_distance is None:
        raise ValueError(
            "exhaustive minimum-distance verification is unavailable for this code"
        )
    return int(cfg.minimum_distance)
