# -*- coding: utf-8 -*-
import heapq
import math
import random
import numpy as np
from .svim import SVIM
DATASET_META = {'syn65k': {'domain_size': 65536, 'user_total': 100000}, 'syn130k': {'domain_size': 131073, 'user_total': 100000}, 'movielens': {'domain_size': 209163, 'user_total': 162414}, 'kosarak': {'domain_size': 41270, 'user_total': 990002}, 'userbehavior1300': {'domain_size': 528150, 'user_total': 863327}, 'ifttt': {'domain_size': 125316, 'user_total': 100000}}

def resolve_dataset_meta(dataname, domain_size=None, user_total=None):
    if domain_size is not None and user_total is not None:
        return (domain_size, user_total)
    key = dataname.lower()
    if key in DATASET_META:
        meta = DATASET_META[key]
        domain_size = meta['domain_size'] if domain_size is None else domain_size
        user_total = meta['user_total'] if user_total is None else user_total
        return (domain_size, user_total)
    known = ', '.join(sorted(DATASET_META.keys()))
    raise ValueError(f"Unknown dataname '{dataname}'. Please provide domain_size and user_total, or add the dataset to DATASET_META. Known datasets: {known}")

def parse_depth_config(raw_value, cast_type=float):
    if raw_value is None:
        return None
    raw = str(raw_value).strip()
    if raw == '':
        return None
    values = []
    for token in raw.split(','):
        token = token.strip()
        if token:
            values.append(cast_type(token))
    return values if values else None

def resolve_depth_value(depth, by_depth_values, default_value, cast_type=float):
    if by_depth_values is not None and len(by_depth_values) > 0:
        idx = min(depth, len(by_depth_values) - 1)
        return cast_type(by_depth_values[idx])
    return cast_type(default_value)

class BSSHHI:

    def __init__(self, data):
        self.data = data

    def encode_items(self, data, b):
        encoded = []
        for user_items in data.data:
            if not user_items:
                encoded.append([])
                continue
            item = random.choice(user_items)
            binary = format(item, f'0{b}b')[-b:]
            indices = [i for i, bit in enumerate(binary[::-1]) if bit == '1']
            encoded.append(indices)
        return encoded

    def clone_data_obj(self, data):
        cloned = type(data).__new__(type(data))
        cloned.__dict__ = data.__dict__.copy()
        cloned.data = [list(items) for items in data.data]
        return cloned

    def update_user_items_with_new_ids(self, data, new_item_ids_list, dict_size):
        data.data = [new_ids for new_ids in new_item_ids_list]
        data.dict_size = dict_size
        data.user_use = len(data.data)
        return data

    def build_encoded_data_subset(self, data, encoded_items, selected_indices, b):
        subset_encoded = [encoded_items[i] for i in selected_indices]
        data_subset = self.clone_data_obj(data)
        return self.update_user_items_with_new_ids(data_subset, subset_encoded, b - 1)

    def build_conditional_report_subset(self, data, encoded_items, selected_indices, fixed_bits, b):
        match_token = b
        dummy_token = b + 1
        reports = []
        for user_idx in selected_indices:
            user_bits = set(encoded_items[user_idx])
            matches = all(((bit_idx in user_bits) == bool(bit_value) for bit_idx, bit_value in fixed_bits.items()))
            if matches:
                report = [match_token]
                report.extend((bit_idx for bit_idx in user_bits if bit_idx not in fixed_bits))
                reports.append(report)
            else:
                reports.append([dummy_token])
        data_subset = self.clone_data_obj(data)
        self.update_user_items_with_new_ids(data_subset, reports, dummy_token)
        data_subset.dummy_item_id = dummy_token
        return data_subset

    def estimate_bit_frequencies(self, encoded_data, epsilon, b):
        if len(encoded_data.data) == 0:
            return (list(range(b)), np.full(b, 0.5, dtype=float))
        sv = SVIM(encoded_data, top_k=b, epsilon=epsilon)
        bit_list, bit_freq = sv.find_singleton_origin(len(encoded_data.data))
        bit_freq = np.array(bit_freq, dtype=float) / max(len(encoded_data.data), 1)
        freq_map = np.full(b, 0.5, dtype=float)
        for bit, freq in zip(bit_list, bit_freq):
            if 0 <= bit < b:
                freq_map[bit] = float(np.clip(freq, 0.0, 1.0))
        return (list(range(b)), freq_map)

    def estimate_conditional_bit_frequencies(self, data, encoded_items, selected_indices, fixed_bits, epsilon, b):
        if not selected_indices:
            return (list(range(b)), np.full(b, 0.5, dtype=float))
        report_data = self.build_conditional_report_subset(data, encoded_items, selected_indices, fixed_bits, b)
        match_token = b
        sv = SVIM(report_data, top_k=b + 1, epsilon=epsilon)
        token_list, token_freq = sv.find_singleton_origin(len(report_data.data))
        token_freq = np.asarray(token_freq, dtype=float)
        freq_by_token = {int(token): float(freq) for token, freq in zip(token_list, token_freq)}
        match_estimate = freq_by_token.get(match_token, 0.0)
        freq_map = np.full(b, 0.5, dtype=float)
        if match_estimate > 0:
            for bit_idx in range(b):
                if bit_idx in fixed_bits:
                    freq_map[bit_idx] = float(fixed_bits[bit_idx])
                else:
                    joint_estimate = freq_by_token.get(bit_idx, 0.0)
                    freq_map[bit_idx] = float(np.clip(joint_estimate / match_estimate, 0.0, 1.0))
        else:
            for bit_idx, bit_value in fixed_bits.items():
                freq_map[bit_idx] = float(bit_value)
        return (list(range(b)), freq_map)

    def top_k_binary_numbers(self, bit_list, bit_freq, b, k, fixed_bits=None):
        fixed_bits = fixed_bits or {}
        bit_prob_map = {bit: freq for bit, freq in zip(bit_list, bit_freq)}
        base_val = 0
        base_prob = 1.0
        changes = []
        for i in range(b):
            if i in fixed_bits:
                p1 = 1.0 if fixed_bits[i] == 1 else 0.0
                if p1 == 1.0:
                    base_val |= 1 << i
                continue
            else:
                p1 = float(np.clip(bit_prob_map.get(i, 0.5), 0.0, 1.0))
            p0 = 1.0 - p1
            if p1 >= p0:
                best_p = p1
                worse_p = p0
                base_val |= 1 << i
            else:
                best_p = p0
                worse_p = p1
            base_prob *= best_p
            factor = worse_p / best_p if best_p > 0 else 0.0
            changes.append((factor, i))
        changes.sort(key=lambda x: x[0], reverse=True)
        pq = [(-base_prob, -1, base_val, base_prob)]
        result = []
        visited = set()
        while len(result) < k and pq:
            neg_p, idx, val, parent_p = heapq.heappop(pq)
            prob = -neg_p
            if val in visited:
                continue
            visited.add(val)
            result.append((prob, format(val, f'0{b}b')))
            if idx + 1 < len(changes):
                next_factor, next_bit_idx = changes[idx + 1]
                new_prob = prob * next_factor
                new_val = val ^ 1 << next_bit_idx
                heapq.heappush(pq, (-new_prob, idx + 1, new_val, prob))
                if idx != -1:
                    prev_bit_idx = changes[idx][1]
                    replace_prob = parent_p * next_factor
                    replace_val = val ^ 1 << prev_bit_idx ^ 1 << next_bit_idx
                    heapq.heappush(pq, (-replace_prob, idx + 1, replace_val, parent_p))
        binary_list = [binary for _, binary in result]
        freq_list = [prob for prob, _ in result]
        return (binary_list, freq_list)

    def to_log_scores(self, scores):
        return [float('-inf') if score <= 0 else float(np.log(score)) for score in scores]

    def apply_soft_candidate_threshold(self, decimal_list, binary_list, score_list, base_k, margin_log=0.0):
        if not decimal_list:
            return ([], [], [], [])
        base_k = max(1, min(int(base_k), len(decimal_list)))
        log_scores = self.to_log_scores(score_list)
        kth_log = log_scores[base_k - 1]
        keep_end = base_k
        while keep_end < len(decimal_list):
            if log_scores[keep_end] < kth_log - margin_log:
                break
            keep_end += 1
        return (decimal_list[:keep_end], binary_list[:keep_end], score_list[:keep_end], log_scores[:keep_end])

    def compute_binary_probability(self, value, bit_freq, b, fixed_bits=None):
        fixed_bits = fixed_bits or {}
        prob = 1.0
        for bit_idx in range(b):
            if bit_idx in fixed_bits:
                p1 = 1.0 if fixed_bits[bit_idx] == 1 else 0.0
            else:
                p1 = float(np.clip(bit_freq[bit_idx], 0.0, 1.0))
            p0 = 1.0 - p1
            prob *= p1 if value >> bit_idx & 1 else p0
        return prob

    def compute_binary_log_probability(self, value, bit_freq, b, fixed_bits=None):
        prob = self.compute_binary_probability(value, bit_freq, b, fixed_bits=fixed_bits)
        return float('-inf') if prob <= 0 else float(np.log(prob))

    def build_pattern_from_fixed_bits(self, b, fixed_bits):
        pattern = ['x'] * b
        for bit_idx, fixed_value in fixed_bits.items():
            if 0 <= bit_idx < b:
                pattern[b - 1 - bit_idx] = str(int(fixed_value))
        return ''.join(pattern)

    def rank_split_bits_uniform(self, node_bit_freq, candidate_split_bits, select_count):
        if not candidate_split_bits:
            return []
        bit_scores = []
        for bit_idx in candidate_split_bits:
            p1 = float(np.clip(node_bit_freq[bit_idx], 0.0, 1.0))
            bias = abs(p1 - 0.5)
            bit_scores.append((bias, bit_idx))
        ranked = sorted(bit_scores, key=lambda x: (x[0], x[1]), reverse=True)
        return [bit_idx for _, bit_idx in ranked[:select_count]]

    def init_matrix(self, group1, epsilon, topk, b, significant_bit_count=2, global_user_ratio=0.4, fixed_bit_one_ratio=0.5, global_candidate_k=None, conditional_candidate_k=None, conditional_candidate_k_by_depth=None, max_split_order=1, node_user_ratio=0.5, significant_bits_by_depth=None, user_ratio_by_depth=None, soft_margin_log=0.0, soft_search_factor=3):
        global_candidate_k = topk if global_candidate_k is None else int(global_candidate_k)
        conditional_candidate_k = topk if conditional_candidate_k is None else int(conditional_candidate_k)
        significant_bit_count = max(1, min(int(significant_bit_count), b))
        max_split_order = max(1, int(max_split_order))
        node_user_ratio = float(np.clip(node_user_ratio, 0.0, 1.0))
        fixed_bit_one_ratio = float(np.clip(fixed_bit_one_ratio, 0.0, 1.0))
        if significant_bits_by_depth is not None:
            significant_bits_by_depth = [max(1, min(int(v), b)) for v in significant_bits_by_depth]
        if user_ratio_by_depth is not None:
            user_ratio_by_depth = [float(np.clip(v, 0.0, 1.0)) for v in user_ratio_by_depth]
        encoded_all_users = self.encode_items(group1, b)
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
            est_ratio = resolve_depth_value(depth=depth, by_depth_values=user_ratio_by_depth, default_value=global_user_ratio if depth == 0 else node_user_ratio, cast_type=float)
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
            if fixed_bits:
                node_bit_list, node_bit_freq = self.estimate_conditional_bit_frequencies(group1, encoded_all_users, est_indices, fixed_bits, epsilon, b)
            else:
                node_data = self.build_encoded_data_subset(group1, encoded_all_users, est_indices, b)
                node_bit_list, node_bit_freq = self.estimate_bit_frequencies(node_data, epsilon, b)
            if depth == 0:
                candidate_k = max(1, global_candidate_k)
            elif conditional_candidate_k_by_depth is not None and len(conditional_candidate_k_by_depth) > 0:
                depth_idx = min(depth - 1, len(conditional_candidate_k_by_depth) - 1)
                candidate_k = max(1, int(conditional_candidate_k_by_depth[depth_idx]))
            else:
                candidate_k = max(1, conditional_candidate_k)
            search_k = max(candidate_k, int(candidate_k * max(1, soft_search_factor)))
            node_bin_raw, node_prob_raw = self.top_k_binary_numbers(node_bit_list, node_bit_freq, b, k=search_k, fixed_bits=fixed_bits)
            node_decimal_raw = [int(binary, 2) for binary in node_bin_raw]
            node_decimal_list, node_bin_list, node_prob_list, node_log_scores = self.apply_soft_candidate_threshold(node_decimal_raw, node_bin_raw, node_prob_raw, base_k=candidate_k, margin_log=float(max(0.0, soft_margin_log)))
            candidate_split_bits = [i for i in range(b) if i not in fixed_bits]
            depth_sig_bits = resolve_depth_value(depth=depth, by_depth_values=significant_bits_by_depth, default_value=significant_bit_count, cast_type=int)
            depth_sig_bits = max(1, min(int(depth_sig_bits), b))
            selected_bits = self.rank_split_bits_uniform(node_bit_freq, candidate_split_bits, depth_sig_bits)
            node_results.append({'node_id': node['node_id'], 'parent_id': node['parent_id'], 'depth': depth, 'fixed_bits': {int(k): int(v) for k, v in fixed_bits.items()}, 'pattern': self.build_pattern_from_fixed_bits(b, fixed_bits), 'split_bit': node['split_bit'], 'split_value': node['split_value'], 'pool_user_count': len(pool_indices), 'estimate_user_count': len(est_indices), 'remaining_user_count': len(remain_indices), 'estimate_ratio': est_ratio, 'depth_significant_bit_count': depth_sig_bits, 'significant_bits': [int(i) for i in selected_bits], 'bit_list': node_bit_list, 'bit_freq': node_bit_freq.tolist(), 'binary_candidates': node_bin_list, 'decimal_candidates': node_decimal_list, 'candidate_scores': node_prob_list, 'candidate_log_scores': node_log_scores, 'base_candidate_k': candidate_k})
            if not can_expand or not selected_bits or (not remain_indices):
                continue
            split_bit_groups = np.array_split(remain_indices, len(selected_bits))
            for sig_bit, bit_group in zip(selected_bits, split_bit_groups):
                bit_group = list(bit_group)
                if not bit_group:
                    continue
                target_one_count = int(len(bit_group) * fixed_bit_one_ratio)
                child_groups = [(1, bit_group[:target_one_count]), (0, bit_group[target_one_count:])]
                for fixed_value, user_subset in child_groups:
                    child_fixed = dict(fixed_bits)
                    child_fixed[int(sig_bit)] = int(fixed_value)
                    queue.append({'node_id': next_node_id, 'parent_id': node['node_id'], 'depth': depth + 1, 'fixed_bits': child_fixed, 'split_bit': int(sig_bit), 'split_value': int(fixed_value), 'pool_indices': list(user_subset)})
                    next_node_id += 1
        root_result = next((node for node in node_results if node['depth'] == 0), None)
        if root_result is None:
            root_result = {'bit_list': list(range(b)), 'bit_freq': [0.5] * b, 'binary_candidates': [], 'decimal_candidates': [], 'candidate_scores': [], 'candidate_log_scores': [], 'significant_bits': [], 'estimate_user_count': 0}
        bit_list = root_result['bit_list']
        bit_freq = np.array(root_result['bit_freq'], dtype=float)
        candidate_bin_list, candidate_score_list = self.top_k_binary_numbers(bit_list, bit_freq, b, k=topk ** 2, fixed_bits={})
        candidate_id_list = [int(binary, 2) for binary in candidate_bin_list]
        ordered_candidate_pairs = sorted(zip(candidate_id_list, candidate_score_list), key=lambda x: (-x[1], x[0]))
        ordered_candidate_ids = [item for item, _ in ordered_candidate_pairs]
        ordered_candidate_scores = [score for _, score in ordered_candidate_pairs]
        s0 = np.array(ordered_candidate_scores).reshape(topk, topk)
        stage1_state = {'global_user_count': root_result['estimate_user_count'], 'global_bit_list': bit_list, 'global_bit_freq': bit_freq.tolist(), 'global_binary_candidates': root_result['binary_candidates'], 'global_decimal_candidates': root_result['decimal_candidates'], 'global_candidate_scores': root_result['candidate_scores'], 'global_candidate_log_scores': root_result['candidate_log_scores'], 'significant_bits': [int(i) for i in root_result['significant_bits']], 'fixed_bit_one_ratio': fixed_bit_one_ratio, 'max_split_order': max_split_order, 'node_user_ratio': node_user_ratio, 'significant_bits_by_depth': significant_bits_by_depth, 'user_ratio_by_depth': user_ratio_by_depth, 'soft_margin_log': float(max(0.0, soft_margin_log)), 'soft_search_factor': int(max(1, soft_search_factor)), 'conditional_results': [node for node in node_results if node['depth'] > 0], 'node_results': node_results}
        return (s0, ordered_candidate_ids, bit_list, stage1_state)

    def split_data_by_ratio(self, data, ratios, shuffle=True):
        if abs(sum(ratios) - 1.0) >= 1e-06:
            raise ValueError('The ratio list must sum to 1.')
        if len(ratios) not in (2, 3):
            raise ValueError('The ratio list must contain 2 or 3 values.')
        indices = list(range(len(data.data)))
        if shuffle:
            random.shuffle(indices)
        sizes = [int(len(indices) * r) for r in ratios]
        sizes[-1] = len(indices) - sum(sizes[:-1])
        groups = []
        start = 0
        for size in sizes:
            end = start + size
            selected_indices = indices[start:end]
            group_data = type(data).__new__(type(data))
            group_data.__dict__ = data.__dict__.copy()
            group_data.data = [data.data[i] for i in selected_indices]
            groups.append(group_data)
            start = end
        return groups

    def perturb_vector(self, x, r, alpha):
        norm_x = np.linalg.norm(x)
        clip_radius = float(r)
        if clip_radius <= 0:
            raise ValueError('The clipping radius must be positive.')
        if norm_x == 0:
            x_tilde = np.zeros_like(x)
            clipped_norm = 0.0
        else:
            clipped_norm = min(float(norm_x), clip_radius)
            prob_pos = 0.5 + clipped_norm / (2 * clip_radius)
            if np.random.rand() < prob_pos:
                x_tilde = clip_radius * x / norm_x
            else:
                x_tilde = -(clip_radius * x) / norm_x
        pi_alpha = np.exp(alpha) / (np.exp(alpha) + 1)
        t = np.random.rand() < pi_alpha
        dimension = len(x)
        log_sphere_factor = math.log(dimension) + 0.5 * math.log(math.pi) + math.lgamma((dimension + 1) / 2) - math.log(2.0) - math.lgamma(dimension / 2 + 1)
        sphere_factor = math.exp(log_sphere_factor)
        scale = clip_radius * (np.exp(alpha) + 1) / (np.exp(alpha) - 1) * sphere_factor
        z = np.random.randn(len(x))
        z = scale * z / np.linalg.norm(z)
        if t:
            if np.dot(z, x_tilde) <= 0:
                z = -z
        elif np.dot(z, x_tilde) > 0:
            z = -z
        return z

    def svd_iteration(self, S0, user_data_list, sorted_indices, d, domain_size, iterations=5, epsilon=5.0, mode='sum', theta=0.9):

        def choose_rank(singular_values):
            if mode == 'square_sum':
                total = float(np.sum(singular_values ** 2))
                partial = 0.0
                for rank, value in enumerate(singular_values, start=1):
                    partial += float(value ** 2)
                    if partial > total * theta:
                        return rank
                return len(singular_values)
            total = float(np.sum(singular_values))
            partial = 0.0
            for rank, value in enumerate(singular_values, start=1):
                partial += float(value)
                if partial > total * theta:
                    return rank
            return len(singular_values)
        S = S0.copy()
        _, singular_values, _ = np.linalg.svd(S, full_matrices=False)
        rank = choose_rank(singular_values)
        n_users = len(user_data_list)
        batch_size = max(1, n_users // max(1, iterations))
        for iteration in range(iterations):
            start_idx = iteration * batch_size
            end_idx = (iteration + 1) * batch_size if iteration < iterations - 1 else n_users
            current_user_data = user_data_list[start_idx:end_idx]
            if not current_user_data:
                continue
            L, _, Rt = np.linalg.svd(S, full_matrices=False)
            Lr = np.asarray(L[:, :rank])
            Rr = np.asarray(Rt[:rank, :])
            sum_vector = np.zeros(rank * rank)
            count = 0
            for user_items in current_user_data:
                user_items_set = set(user_items)
                selected_values = np.array([1 if idx <= domain_size and idx in user_items_set else 0 for idx in sorted_indices])
                sub_matrix = selected_values.reshape((d, d))
                user_proj = Lr.T @ sub_matrix @ Rr.T
                perturbed_vec = self.perturb_vector(np.asarray(user_proj).flatten(), r=1, alpha=epsilon)
                sum_vector += perturbed_vec
                count += 1
            if count == 0:
                continue
            avg_vector = sum_vector / count
            S_new = avg_vector.reshape(rank, rank)
            S = Lr @ S_new @ Rr
        return np.asarray(S)

    def get_topd_original_indices(self, S_iter, k_list, d, invalid_ids=None):
        flattened_S = S_iter.flatten()
        ranked_indices = np.argsort(flattened_S)[::-1]
        invalid_ids = set() if invalid_ids is None else set(invalid_ids)
        results = []
        seen = set()
        for idx in ranked_indices:
            candidate = k_list[idx]
            if candidate in invalid_ids or candidate in seen:
                continue
            seen.add(candidate)
            results.append(candidate)
            if len(results) >= d:
                break
        return results

    def remove_non_top_items(self, data, top_d_list):
        top_d_set = set(top_d_list)
        item_mapping = {old_id: new_id for new_id, old_id in enumerate(top_d_list)}
        reverse_mapping = {new_id: old_id for new_id, old_id in enumerate(top_d_list)}
        new_data = []
        for user_items in data.data:
            filtered = [item_mapping[item] for item in user_items if item in top_d_set]
            new_data.append(filtered if filtered else [-1])
        data.data = new_data
        data.dict_size = len(top_d_list)
        data.user_use = len(new_data)
        data.id_mapping = item_mapping
        data.reverse_mapping = reverse_mapping
        return (data, data.user_use)

    def recover_items_from_top_list(self, mapped_list, top_d_list):
        reverse_mapping = {new_id: original_id for new_id, original_id in enumerate(top_d_list)}
        return [reverse_mapping[i] for i in mapped_list if i in reverse_mapping]

    def collect_stage1_candidate_union(self, stage1_details):
        candidate_union = set()
        for node in stage1_details['node_results']:
            candidate_union.update(node['decimal_candidates'])
        return sorted(candidate_union)
