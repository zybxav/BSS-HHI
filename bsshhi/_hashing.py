# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

@njit(cache=True, inline='always')
def _xxh32_short(data, length, hash_key):
    mask = 4294967295
    h = hash_key + 374761393 + length & mask
    pos = 0
    while pos + 4 <= length:
        word = int(data[pos]) | int(data[pos + 1]) << 8 | int(data[pos + 2]) << 16 | int(data[pos + 3]) << 24
        h = h + word * 3266489917 & mask
        h = (h << 17 | h >> 15) & mask
        h = h * 668265263 & mask
        pos += 4
    while pos < length:
        h = h + int(data[pos]) * 374761393 & mask
        h = (h << 11 | h >> 21) & mask
        h = h * 2654435761 & mask
        pos += 1
    h ^= h >> 15
    h = h * 2246822519 & mask
    h ^= h >> 13
    h = h * 3266489917 & mask
    h ^= h >> 16
    return h

def fast_aggregate(noisy_samples, domain, g, p, q):
    n = len(noisy_samples)
    strings = [str(v).encode('ascii') for v in range(domain)]
    width = max(map(len, strings), default=1)
    if width >= 16:
        raise ValueError('short-hash accelerator supports IDs with at most 15 digits')
    data = np.zeros((domain, width), dtype=np.uint8)
    lengths = np.array([len(s) for s in strings], dtype=np.int64)
    for i, s in enumerate(strings):
        data[i, :len(s)] = np.frombuffer(s, dtype=np.uint8)
    reports = np.array([x[0] for x in noisy_samples], dtype=np.int64)
    hash_keys = np.array([x[1] for x in noisy_samples], dtype=np.int64)
    distinct, inverse = np.unique(hash_keys, return_inverse=True)
    table = np.bincount(inverse * int(g) + reports, minlength=len(distinct) * int(g)).reshape(len(distinct), int(g))
    est = _counts_grouped(data, lengths, table, distinct, int(g))
    a = 1.0 / (p - q)
    b = n * q / (p - q)
    return a * est - b

@njit(cache=True)
def _counts_grouped(data, lengths, table, hash_keys, g):
    counts = np.zeros(len(lengths), dtype=np.int32)
    reciprocal = 1.0 / g
    for v in range(len(lengths)):
        total = 0
        for i in range(len(hash_keys)):
            h = _xxh32_short(data[v], lengths[v], hash_keys[i])
            quotient = int(h * reciprocal)
            remainder = h - quotient * g
            if remainder >= g:
                remainder -= g
            if remainder < 0:
                remainder += g
            total += table[i, remainder]
        counts[v] = total
    return counts
