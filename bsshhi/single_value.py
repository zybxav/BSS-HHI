# -*- coding: utf-8 -*-
"""Single-valued BSS-HHI: candidate reconstruction and independent OLH ranking."""
import math
import numpy as np
from numba import njit
from .algorithm import BSSHHI
from .data import Data

def _prime(value):
    value = max(2, int(value))
    while True:
        if value == 2 or (value % 2 and all((value % d for d in range(3, int(math.sqrt(value)) + 1, 2)))):
            return value
        value += 1

@njit(cache=True)
def _decode(a, b, reports, queries, prime, g):
    counts = np.zeros(len(queries), dtype=np.int64)
    for j in range(len(queries)):
        for i in range(len(a)):
            if (a[i] * queries[j] + b[i]) % prime % g == reports[i]:
                counts[j] += 1
    return counts

def olh(values, domain, epsilon, queries):
    rng = np.random.default_rng()
    g = max(2, int(round(math.exp(epsilon))) + 1)
    prime = _prime(domain)
    groups = np.arange(len(values), dtype=np.int64)
    rng.shuffle(groups)
    a = rng.integers(1, prime, size=len(values), dtype=np.int64)
    b = rng.integers(0, prime, size=len(values), dtype=np.int64)
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    true = (a[groups] * values + b[groups]) % prime % g
    keep = rng.random(len(values)) < p
    offset = rng.integers(1, g, size=len(values))
    reports = np.where(keep, true, (true + offset) % g)
    order = np.argsort(groups)
    reports = reports[order]
    matches = _decode(a, b, reports, np.asarray(queries, dtype=np.int64), prime, g)
    quotient, remainder = divmod(prime, g)
    collision = (remainder * (quotient + 1) * quotient + (g - remainder) * quotient * (quotient - 1)) / (prime * (prime - 1))
    q = 1 / (math.exp(epsilon) + g - 1)
    background = q + collision * (p - q)
    return (matches - len(values) * background) / (p - background)

def evaluate(values, domain, epsilon, topk, parameters, dataset):
    values = np.asarray(values, dtype=np.int64) - 1
    rng = np.random.default_rng()
    indices = rng.permutation(len(values))
    count = int(0.2 * len(values))
    discover, estimate = (values[indices[:count]], values[indices[count:]])
    data = Data([[int(v) + 1] for v in discover], dataset, domain)
    bss = BSSHHI(data)
    *_, state = bss.init_matrix(data, epsilon=epsilon, topk=int(np.ceil(np.sqrt(parameters['global_candidate_k']))), b=int(math.ceil(math.log2(domain + 1))), global_candidate_k=parameters['global_candidate_k'], conditional_candidate_k=50, conditional_candidate_k_by_depth=[50, 35, 25, 15, 10], max_split_order=parameters['max_split_order'], significant_bits_by_depth=parameters['significant_bits_by_depth'], user_ratio_by_depth=parameters['user_ratio_by_depth'], fixed_bit_one_ratio=0.3, soft_margin_log=0.08, soft_search_factor=2)
    candidates = np.array(sorted({item - 1 for item in bss.collect_stage1_candidate_union(state) if 1 <= item <= domain}), dtype=np.int64)
    estimates = olh(estimate, domain, epsilon, candidates)
    chosen = candidates[np.lexsort((candidates, -estimates))[:topk]]
    counts = np.bincount(values, minlength=domain)
    ids = np.arange(domain)
    truth = ids[np.lexsort((ids, -counts))[:topk]]
    hits = set(chosen) & set(truth)
    return dict(f1=2 * len(hits) / (len(chosen) + len(truth)) if len(chosen) + len(truth) else 0.0)
