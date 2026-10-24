# -*- coding: utf-8 -*-
import random
import numpy as np

def sample_encoded_items(data, b):
    encoded_items = []
    for user_items in data.data:
        if not user_items:
            encoded_items.append([])
            continue
        item = random.choice(user_items)
        if isinstance(item, tuple):
            item = item[0]
        value = int(item)
        binary = format(value, f'0{b}b')[-b:]
        encoded_items.append([idx for idx, bit in enumerate(binary[::-1]) if bit == '1'])
    return encoded_items

def select_split_bits(selector_type, node_bit_freq, candidate_split_bits, select_count):
    if not candidate_split_bits:
        return ([], {})
    select_count = max(1, min(int(select_count), len(candidate_split_bits)))
    selector_type = str(selector_type)
    if selector_type == 'least_imbalanced':
        scored = []
        for bit_idx in candidate_split_bits:
            p1 = float(np.clip(node_bit_freq[bit_idx], 0.0, 1.0))
            scored.append((abs(p1 - 0.5), -bit_idx, bit_idx))
        scored.sort(key=lambda item: (item[0], item[1]))
        selected = [bit_idx for _, _, bit_idx in scored[:select_count]]
        return (selected, {str(bit_idx): float(score) for score, _, bit_idx in scored[:select_count]})
    scored = []
    for bit_idx in candidate_split_bits:
        p1 = float(np.clip(node_bit_freq[bit_idx], 0.0, 1.0))
        scored.append((abs(p1 - 0.5), bit_idx))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [bit_idx for _, bit_idx in scored[:select_count]]
    return (selected, {str(bit_idx): float(score) for score, bit_idx in scored[:select_count]})

def init_matrix_with_selector(module, bss, group1, epsilon, b, selector_type, significant_bit_count=2, global_user_ratio=0.4, fixed_bit_one_ratio=0.5, global_candidate_k=None, conditional_candidate_k=None, conditional_candidate_k_by_depth=None, max_split_order=1, node_user_ratio=0.5, significant_bits_by_depth=None, user_ratio_by_depth=None, soft_margin_log=0.0, soft_search_factor=3, expand_children=True):
    global_candidate_k = int(global_candidate_k)
    conditional_candidate_k = int(conditional_candidate_k or global_candidate_k)
    significant_bit_count = max(1, min(int(significant_bit_count), b))
    max_split_order = max(1, int(max_split_order))
    node_user_ratio = float(np.clip(node_user_ratio, 0.0, 1.0))
    fixed_bit_one_ratio = float(np.clip(fixed_bit_one_ratio, 0.0, 1.0))
    if significant_bits_by_depth is not None:
        significant_bits_by_depth = [max(1, min(int(v), b)) for v in significant_bits_by_depth]
    if user_ratio_by_depth is not None:
        user_ratio_by_depth = [float(np.clip(v, 0.0, 1.0)) for v in user_ratio_by_depth]
    if conditional_candidate_k_by_depth is not None:
        conditional_candidate_k_by_depth = [max(1, int(v)) for v in conditional_candidate_k_by_depth]
    encoded_all_users = sample_encoded_items(group1, b)
    all_indices = list(range(len(group1.data)))
    random.shuffle(all_indices)
    node_results = []
    queue = [{'node_id': 0, 'parent_id': None, 'depth': 0, 'fixed_bits': {}, 'split_bit': None, 'split_value': None, 'pool_indices': all_indices}]
    next_node_id = 1
    while queue:
        node = queue.pop(0)
        pool_indices = list(node['pool_indices'])
        if not pool_indices:
            continue
        random.shuffle(pool_indices)
        depth = int(node['depth'])
        fixed_bits = dict(node['fixed_bits'])
        can_expand = depth < max_split_order
        est_ratio = module.resolve_depth_value(depth=depth, by_depth_values=user_ratio_by_depth, default_value=global_user_ratio if depth == 0 else node_user_ratio, cast_type=float)
        est_ratio = float(np.clip(est_ratio, 0.0, 1.0))
        if not can_expand:
            est_count = len(pool_indices)
        else:
            est_count = int(len(pool_indices) * est_ratio)
            if len(pool_indices) == 1:
                est_count = 1
            else:
                est_count = min(max(est_count, 1), len(pool_indices) - 1)
        est_indices = pool_indices[:est_count]
        remain_indices = pool_indices[est_count:]
        node_data = bss.build_encoded_data_subset(group1, encoded_all_users, est_indices, b)
        node_bit_list, node_bit_freq = bss.estimate_bit_frequencies(node_data, epsilon, b)
        if depth == 0:
            candidate_k = max(1, int(global_candidate_k))
        elif conditional_candidate_k_by_depth is not None and len(conditional_candidate_k_by_depth) > 0:
            depth_idx = min(depth - 1, len(conditional_candidate_k_by_depth) - 1)
            candidate_k = max(1, int(conditional_candidate_k_by_depth[depth_idx]))
        else:
            candidate_k = max(1, int(conditional_candidate_k))
        search_k = max(candidate_k, int(candidate_k * max(1, soft_search_factor)))
        node_bin_raw, node_prob_raw = bss.top_k_binary_numbers(node_bit_list, node_bit_freq, b, k=search_k, fixed_bits=fixed_bits)
        node_decimal_raw = [int(binary, 2) for binary in node_bin_raw]
        node_decimal_list, node_bin_list, node_prob_list, node_log_scores = bss.apply_soft_candidate_threshold(node_decimal_raw, node_bin_raw, node_prob_raw, base_k=candidate_k, margin_log=float(max(0.0, soft_margin_log)))
        candidate_split_bits = [idx for idx in range(b) if idx not in fixed_bits]
        depth_sig_bits = module.resolve_depth_value(depth=depth, by_depth_values=significant_bits_by_depth, default_value=significant_bit_count, cast_type=int)
        depth_sig_bits = max(1, min(int(depth_sig_bits), len(candidate_split_bits) if candidate_split_bits else 1))
        selected_bits, selector_scores = select_split_bits(selector_type, node_bit_freq, candidate_split_bits, depth_sig_bits)
        node_results.append({'depth': depth, 'bit_freq': np.asarray(node_bit_freq, dtype=float).tolist(), 'decimal_candidates': [int(v) for v in node_decimal_list]})
        if not expand_children or not can_expand or (not selected_bits) or (not remain_indices):
            continue
        split_bit_groups = np.array_split(remain_indices, len(selected_bits))
        for sig_bit, bit_group in zip(selected_bits, split_bit_groups):
            bit_group = list(bit_group)
            if not bit_group:
                continue
            one_users = [user_idx for user_idx in bit_group if sig_bit in encoded_all_users[user_idx]]
            zero_users = [user_idx for user_idx in bit_group if sig_bit not in encoded_all_users[user_idx]]
            target_one_count = int(len(bit_group) * fixed_bit_one_ratio)
            target_zero_count = len(bit_group) - target_one_count
            child_groups = [(1, one_users[:min(target_one_count, len(one_users))]), (0, zero_users[:min(target_zero_count, len(zero_users))])]
            for fixed_value, user_subset in child_groups:
                if not user_subset:
                    continue
                child_fixed = dict(fixed_bits)
                child_fixed[int(sig_bit)] = int(fixed_value)
                queue.append({'node_id': int(next_node_id), 'parent_id': int(node['node_id']), 'depth': depth + 1, 'fixed_bits': child_fixed, 'split_bit': int(sig_bit), 'split_value': int(fixed_value), 'pool_indices': list(user_subset)})
                next_node_id += 1
    root_result = next((node for node in node_results if int(node['depth']) == 0), None)
    if root_result is None:
        root_result = {'estimate_user_count': 0, 'bit_list': list(range(b)), 'bit_freq': [0.5] * b, 'binary_candidates': [], 'decimal_candidates': [], 'candidate_scores': [], 'candidate_log_scores': [], 'significant_bits': []}
    return {'global_bit_freq': root_result['bit_freq'], 'global_decimal_candidates': root_result['decimal_candidates'], 'node_results': node_results}
