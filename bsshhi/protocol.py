# -*- coding: utf-8 -*-
"""Candidate reconstruction, low-rank refinement and final identification."""
import math
import numpy as np
from .algorithm import BSSHHI, DATASET_META
from .metrics import MetricsItem
from .svim import SVIM
DECAY = (1.0, 0.7, 0.5, 0.3, 0.2)

def matrix_from_candidates(items, scores, padding):
    side = max(1, int(math.ceil(math.sqrt(max(1, len(items))))))
    missing = side * side - len(items)
    return (np.asarray(list(scores) + [0.0] * missing).reshape(side, side), list(items) + [padding] * missing)

def reconstruct(bss, data, epsilon, parameters, allocation):
    bits = parameters['c']
    capacities = [max(1, int(round(parameters['c0'] * decay))) for decay in DECAY]
    *_, state = bss.init_matrix(data, epsilon=epsilon, topk=int(math.ceil(math.sqrt(parameters['x1']))), b=int(math.ceil(math.log2(DATASET_META[data.data_name]['domain_size'] + 1))), global_candidate_k=parameters['x1'], conditional_candidate_k=capacities[0], conditional_candidate_k_by_depth=capacities if bits else [1], max_split_order=len(bits) if bits else 1, significant_bits_by_depth=bits or [1], user_ratio_by_depth=allocation if bits else None, significant_bit_count=2, global_user_ratio=0.5, node_user_ratio=0.5, fixed_bit_one_ratio=0.45, soft_margin_log=0.05, soft_search_factor=2)
    x1 = list(dict.fromkeys(state['global_decimal_candidates']))
    candidates = bss.collect_stage1_candidate_union(state) if bits else x1
    width = int(math.ceil(math.log2(DATASET_META[data.data_name]['domain_size'] + 1)))
    scored = [(item, float(bss.compute_binary_probability(item, state['global_bit_freq'], width, {}))) for item in candidates]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    items = [item for item, score in scored]
    matrix, padded = matrix_from_candidates(items, [score for item, score in scored], data.dict_size + 1)
    return (x1, items, matrix, padded)

def refine(bss, data, matrix, padded, epsilon, parameters, skip=False):
    if skip:
        return matrix
    return bss.svd_iteration(matrix, data.data, padded, d=matrix.shape[0], domain_size=data.dict_size, iterations=parameters['iterations'], epsilon=epsilon, theta=parameters['theta'])

def retain(bss, matrix, padded, union_size, topk, ratio, domain):
    count = min(union_size, max(topk, int(math.ceil(ratio * union_size))))
    return list(dict.fromkeys(bss.get_topd_original_indices(matrix, padded, count, invalid_ids={domain + 1})))

def estimate(bss, data, candidates, epsilon, topk):
    if not candidates:
        return {}
    data, users = bss.remove_non_top_items(data, candidates)
    if not users:
        return {}
    items, frequencies = SVIM(data, top_k=topk, epsilon=epsilon).find_singleton_origin(users)
    return {(candidates[item],): float(frequency) for item, frequency in zip(items, frequencies) if 0 <= item < len(candidates)}

def scores(data, predictions, topk):
    metrics = MetricsItem(data, topk)
    return dict(f1=metrics.item_f1(predictions), ncr=metrics.item_ncr(predictions), mcr=metrics.cal_mcr(predictions))

def identify(data, epsilon, topk, parameters, allocation, cohorts=(0.1, 0.1, 0.8)):
    bss = BSSHHI(data)
    group1, group2, group3 = bss.split_data_by_ratio(data, cohorts)
    _, union, matrix, padded = reconstruct(bss, group1, epsilon, parameters, allocation)
    refined = refine(bss, group2, matrix, padded, epsilon, parameters)
    candidates = retain(bss, refined, padded, len(union), topk, parameters['rho2'], data.dict_size)
    return estimate(bss, group3, candidates, epsilon, topk)

def validate(parameters, domain):
    if any((int(v) != v or not 1 <= v <= math.ceil(math.log2(domain + 1)) for v in parameters['c'])):
        raise ValueError('c entries must be positive integers within the bit width')
    if len(parameters['c']) > 5:
        raise ValueError('At most five conditional expansion iterations are supported')
    if not 1 <= parameters['x1'] <= domain:
        raise ValueError('Phase 1 capacity is outside the domain')
    if parameters['c0'] < 1:
        raise ValueError('C0 must be positive')
    if not 0 < parameters['rho2'] <= 1 or not 0 < parameters['theta'] <= 1:
        raise ValueError('rho2 and theta must lie in (0,1]')
    if parameters['iterations'] < 1:
        raise ValueError('T must be positive')
