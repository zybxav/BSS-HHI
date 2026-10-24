# -*- coding: utf-8 -*-
import heapq
import random
import numpy as np
from ._case_svim import SVIM

class RelationKernel:

    def __init__(self, data):
        self.data = data

    def encode_items(self, data, b):
        result = []
        for user_items in data.data:
            if not user_items:
                result.append([])
                continue
            item = random.choice(user_items)
            binary = format(item, f'0{b}b')[-b:]
            indices = [i for i, bit in enumerate(binary[::-1]) if bit == '1']
            result.append(indices)
        return result

    def top_k_binary_numbers(self, bit_list, bit_freq, b, k):
        bit_prob_map = {bit: freq for bit, freq in zip(bit_list, bit_freq)}
        result = []
        for num in range(2 ** b):
            binary = format(num, f'0{b}b')
            prob = 1.0
            for i, bit_char in enumerate(reversed(binary)):
                bit_index = i
                freq = bit_prob_map.get(bit_index, 0.5)
                if bit_char == '1':
                    prob *= freq
                else:
                    prob *= 1 - freq
            result.append((prob, binary))
        top_k = heapq.nlargest(k, result)
        binary_list = [binary for _, binary in top_k]
        freq_list = [prob for prob, _ in top_k]
        return (binary_list, freq_list)

    def init_matrix(self, group1, epsilon, topk, b):
        new_group1 = self.encode_items(group1, b)
        data_init = self.update_user_items_with_new_ids(group1, new_group1, b - 1)
        true_item_cand = data_init.true_item_cand(topk)
        sv_init = SVIM(data_init, top_k=b, epsilon=epsilon)
        bit_list, bit_freq = sv_init.find_singleton_origin(len(data_init.data))
        bit_freq = np.array(bit_freq)
        bit_freq /= len(data_init.data)
        k_bin_list, k_freq_list = self.top_k_binary_numbers(bit_list, bit_freq, b, k=topk ** 2)
        k_bin_decimal = [int(b, 2) for b in k_bin_list]
        sorted_k_bin_decimal = sorted(k_bin_decimal)
        sorted_indices = np.argsort(k_bin_decimal)
        k_freq_list = [k_freq_list[i] for i in sorted_indices]
        s0 = np.array(k_freq_list).reshape(topk, topk)
        return (s0, sorted_k_bin_decimal, bit_list)

    def split_data_by_ratio(self, data, ratios, shuffle=True):
        assert abs(sum(ratios) - 1.0) < 1e-06, 'ratios 总和必须为 1'
        total_users = len(data.data)
        indices = list(range(total_users))
        if shuffle:
            random.shuffle(indices)
        sizes = [int(total_users * r) for r in ratios]
        sizes[-1] = total_users - sum(sizes[:-1])
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

    def l2_ball_mechanism(self, vectors, r, alpha):

        def generate_X_tilde(x, r):
            norm_x = np.linalg.norm(x)
            if norm_x == 0:
                return np.zeros_like(x)
            prob_pos = 0.5 + norm_x / (2 * r)
            prob_neg = 0.5 - norm_x / (2 * r)
            if np.random.rand() < prob_pos:
                X_tilde = r * x / norm_x
            else:
                X_tilde = -(r * x) / norm_x
            return X_tilde

        def generate_T(alpha):
            pi_alpha = np.exp(alpha) / (np.exp(alpha) + 1)
            return np.random.rand() < pi_alpha

        def select_Z(T, X_tilde, r, d, alpha):
            B = r * (np.exp(alpha) + 1) / (np.exp(alpha) - 1) * np.sqrt(np.pi * d / 2)
            if T == 1:
                z = np.random.randn(d)
                z = B * z / np.linalg.norm(z)
                if np.dot(z, X_tilde) <= 0:
                    z = -z
                return z
            else:
                z = np.random.randn(d)
                z = B * z / np.linalg.norm(z)
                if np.dot(z, X_tilde) > 0:
                    z = -z
                return z
        perturbed_vectors = []
        for x in vectors:
            X_tilde = generate_X_tilde(x, r)
            T = generate_T(alpha)
            Z = select_Z(T, X_tilde, r, len(x), alpha)
            perturbed_vectors.append(Z)
        return perturbed_vectors

    def get_topd_original_indices(self, S_iter, k_list, d):
        flattened_S = S_iter.flatten()
        top_d_indices = np.argsort(flattened_S)[-d:]
        top_d_k_list_indices = [k_list[i] for i in top_d_indices]
        return top_d_k_list_indices

    def update_user_items_with_new_ids(self, data, new_item_ids_list, dict_size):
        new_data = []
        for user_items, new_ids in zip(data.data, new_item_ids_list):
            updated_items = new_ids
            new_data.append(updated_items)
        data.data = new_data
        data.dict_size = dict_size
        data.user_use = len(new_data)
        return data

    def remove_non_top_items(self, data, top_d_list):
        top_d_set = set(top_d_list)
        item_mapping = {old_id: new_id for new_id, old_id in enumerate(top_d_list)}
        reverse_mapping = {new_id: old_id for new_id, old_id in enumerate(top_d_list)}
        new_data = []
        for user_items in data.data:
            filtered = [item_mapping[item] for item in user_items if item in top_d_set]
            if filtered:
                new_data.append(filtered)
        data.data = new_data
        data.dict_size = len(top_d_list)
        data.user_use = len(new_data)
        data.id_mapping = item_mapping
        data.reverse_mapping = reverse_mapping
        return (data, data.user_use)

    def recover_items_from_top_list(self, mapped_list, top_d_list):
        reverse_mapping = {new_id: original_id for new_id, original_id in enumerate(top_d_list)}
        return [reverse_mapping[i] for i in mapped_list if i in reverse_mapping]
