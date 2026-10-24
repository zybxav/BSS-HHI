# -*- coding: utf-8 -*-
"""BSS-HHI candidate reconstruction for the paper's ID-encoding experiments."""
import math
import numpy as np
from . import _encoding_kernel as algorithm
from ._encoding_kernel import BSSHHI
from .data import Data
from .encoding import init_matrix_with_selector
from ._case_svim import SVIM
from .metrics import MetricsItem
from .protocol import matrix_from_candidates

def partial_permutation(domain, fraction, order):
    mapping = np.arange(domain + 1, dtype=np.int64)
    subset = order[:int(round(domain * fraction))]
    if len(subset) > 1:
        shuffled = np.random.permutation(subset)
        if np.any(shuffled == subset):
            shuffled = np.roll(subset, 1)
        mapping[subset] = shuffled
    return mapping

def csv_list(value, kind=int):
    return [kind(item) for item in str(value).split(',')]

def evaluate(data, epsilon, topk, parameters, mapping):
    inverse = np.empty_like(mapping)
    inverse[mapping] = np.arange(len(mapping))
    encoded = Data([[int(mapping[item]) if item >= 0 else item for item in row] for row in data.data], data.data_name, data.dict_size)
    bss = BSSHHI(encoded)
    group1, group2, group3 = bss.split_data_by_ratio(encoded, [0.1, 0.1, 0.8])
    full = parameters['search_mode'] != 'global_only'
    state = init_matrix_with_selector(algorithm, bss, group1, epsilon, int(math.ceil(math.log2(data.dict_size + 1))), parameters['selector_type'] if full else 'imbalance', significant_bit_count=2, global_user_ratio=float(parameters['global_user_ratio']), fixed_bit_one_ratio=float(parameters['fixed_bit_one_ratio']), global_candidate_k=int(parameters['global_candidate_k']), conditional_candidate_k=csv_list(parameters['conditional_candidate_k_by_depth'])[0] if full else 1, conditional_candidate_k_by_depth=csv_list(parameters['conditional_candidate_k_by_depth']) if full else None, max_split_order=int(parameters['max_split_order']), node_user_ratio=float(parameters['node_user_ratio']), significant_bits_by_depth=csv_list(parameters['significant_bits_by_depth']) if full else None, user_ratio_by_depth=csv_list(parameters['user_ratio_by_depth'], float) if full else None, soft_margin_log=0.0, soft_search_factor=3, expand_children=full)
    union = list(dict.fromkeys((item for node in state['node_results'] for item in node['decimal_candidates']))) if full else list(dict.fromkeys(state['global_decimal_candidates']))
    items = [int(item) for item in union if 1 <= item <= data.dict_size]
    width = int(math.ceil(math.log2(data.dict_size + 1)))
    scores = {item: float(bss.compute_binary_probability(item, state['global_bit_freq'], width, {})) for item in items}
    cap = int(parameters['final_candidate_cap'])
    if cap > 0 and len(items) > cap:
        items = sorted(items, key=lambda item: (-scores[item], item))[:cap]
    matrix, padded = matrix_from_candidates(items, [scores[item] for item in items], data.dict_size + 1)
    if parameters['phase3_method'] == 'score_filter':
        ordered = sorted(items, key=lambda item: (-scores[item], item))
    else:
        refined = bss.svd_iteration(matrix, group2.data, padded, d=matrix.shape[0], domain_size=data.dict_size, iterations=int(parameters['iterations']), epsilon=epsilon, theta=float(parameters['theta']))
        ordered = list(dict.fromkeys(bss.get_topd_original_indices(refined, padded, len(items), invalid_ids={data.dict_size + 1})))
    count = min(len(items), max(topk, int(math.ceil(float(parameters['output_ratio_of_union']) * len(items)))))
    candidates = ordered[:count]
    if not candidates:
        return dict(f1=0.0)
    index = {item: i for i, item in enumerate(candidates)}
    group3.data = [[index[item] for item in row if item in index] for row in group3.data]
    group3.data = [row for row in group3.data if row]
    group3.dict_size = len(candidates)
    group3.user_use = len(group3.data)
    if not group3.data:
        return dict(f1=0.0)
    result, freq = SVIM(group3, top_k=topk, epsilon=epsilon).find_singleton_origin(len(group3.data))
    predictions = {(int(inverse[candidates[item]]),): float(value) for item, value in zip(result, freq) if 0 <= item < len(candidates)}
    return dict(f1=MetricsItem(data, topk).item_f1(predictions))
