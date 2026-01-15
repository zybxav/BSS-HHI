import heapq
import random
from datetime import datetime
import numpy as np
import sys
import os

sys.path.append("..")

from svim import SVIM, hit_rate_curve
from data_rm import Data
from metrics_item import MetricsItem




class BSSHHI:
    def __init__(self, data):
        self.data = data


    def encode_items(self,data, b):
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
        base_val = 0
        base_prob = 1.0
        changes = [] # (factor, bit_index)

        for i in range(b):
            p1 = bit_prob_map.get(i, 0.5)
            p0 = 1.0 - p1

            if p1 >= p0:
                best_p = p1
                worse_p = p0
                base_val |= (1 << i)
            else:
                best_p = p0
                worse_p = p1
            
            base_prob *= best_p

            if best_p > 0:
                factor = worse_p / best_p
            else:
                factor = 0.0
            
            changes.append((factor, i))

        changes.sort(key=lambda x: x[0], reverse=True)
        
        pq = [(-base_prob, -1, base_val, base_prob)]
        
        result = []
        
        while len(result) < k and pq:
            neg_p, idx, val, parent_p = heapq.heappop(pq)
            prob = -neg_p
            
            binary_str = format(val, f'0{b}b')
            result.append((prob, binary_str))
            
            if idx + 1 < len(changes):
                next_factor, next_bit_idx = changes[idx+1]

                new_prob = prob * next_factor
                new_val = val ^ (1 << next_bit_idx)
                heapq.heappush(pq, (-new_prob, idx+1, new_val, prob))

                if idx != -1:
                    prev_bit_idx = changes[idx][1]
                    # (prob / prev_factor) * next_factor = parent_p * next_factor
                    replace_prob = parent_p * next_factor
                    replace_val = val ^ (1 << prev_bit_idx) ^ (1 << next_bit_idx)
                    heapq.heappush(pq, (-replace_prob, idx+1, replace_val, parent_p))

        binary_list = [binary for _, binary in result]
        freq_list = [prob for prob, _ in result]

        return binary_list, freq_list

    def init_matrix(self, group1, epsilon, topk, b):
        new_group1 = self.encode_items(group1,b)
        data_init = self.update_user_items_with_new_ids(group1,new_group1,b-1)
        true_item_cand = data_init.true_item_cand(topk)
        sv_init = SVIM(data_init, top_k=b,epsilon=epsilon)
        bit_list, bit_freq= sv_init.find_singleton_origin(len(data_init.data))
        bit_freq = np.array(bit_freq)

        bit_freq /= (len(data_init.data))
        k_bin_list, k_freq_list = self.top_k_binary_numbers(bit_list, bit_freq, b, k=topk**2)
        k_bin_decimal = [int(b, 2) for b in k_bin_list]
        sorted_k_bin_decimal = sorted(k_bin_decimal)
        sorted_indices = np.argsort(k_bin_decimal)
        k_freq_list = [k_freq_list[i] for i in sorted_indices]

        s0 = np.array(k_freq_list).reshape(topk,topk)
        return s0,sorted_k_bin_decimal,bit_list


    def extract_submatrices_from_flat_indices(self,big_matrices, index_list, d):
        sub_matrices = []
        index_list = [int(b, 2) for b in index_list]
        for big_matrix in big_matrices:
            flat = big_matrix.flatten()
            values = flat[index_list]
            sub_matrix = values.reshape((d, d))
            sub_matrices.append(sub_matrix)

        return sub_matrices

    def split_data_by_ratio(self, data, ratios, shuffle=True):
        assert abs(sum(ratios) - 1.0) < 1e-6, "sum of ratios must=1 "
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
                X_tilde = (r * x) / norm_x
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

    def perturb_vector(self, x, r, alpha):

        def generate_X_tilde(x, r):
            norm_x = np.linalg.norm(x)
            if norm_x == 0:
                return np.zeros_like(x)
            
            prob_pos = 0.5 + norm_x / (2 * r)
            
            if np.random.rand() < prob_pos:
                X_tilde = (r * x) / norm_x
            else:
                X_tilde = -(r * x) / norm_x
            return X_tilde

        def generate_T(alpha):
            pi_alpha = np.exp(alpha) / (np.exp(alpha) + 1)
            return np.random.rand() < pi_alpha

        def select_Z(T, X_tilde, r, d, alpha):
            B = r * (np.exp(alpha) + 1) / (np.exp(alpha) - 1) * np.sqrt(np.pi * d / 2)
            z = np.random.randn(d)
            z = B * z / np.linalg.norm(z)
            
            if T == 1:
                if np.dot(z, X_tilde) <= 0:
                    z = -z
            else:
                if np.dot(z, X_tilde) > 0:
                    z = -z
            return z

        X_tilde = generate_X_tilde(x, r)
        T = generate_T(alpha)
        Z = select_Z(T, X_tilde, r, len(x), alpha)
        return Z

    def build_submatrices_from_batch(self, user_items_list, sorted_indices, d, domain_size):


        sub_matrices = []
        for user_items in user_items_list:
            user_vector = np.zeros(domain_size+1, dtype=int)
            for item in user_items:
                if 0 <= item < domain_size:
                    user_vector[item] = 1

            selected_values = np.array([
                user_vector[i] if i <= domain_size else 0
                for i in sorted_indices
            ])

            sub_matrix = selected_values.reshape((d, d))
            sub_matrices.append(sub_matrix)

        return sub_matrices

    def build_user_matrix(self, data_obj, matrix_size):
        domain_size = data_obj.dict_size
        padded_size = matrix_size ** 2

        user_matrices = []

        for user_items in data_obj.data:
            user_vector = np.zeros(padded_size, dtype=int)

            for item in user_items:
                if 0 <= item < domain_size:
                    user_vector[item] = 1

            user_matrix = user_vector.reshape((matrix_size, matrix_size))
            user_matrices.append(user_matrix)

        return user_matrices, matrix_size

    def svd_iteration(self, S0, user_data_list, sorted_indices, d, domain_size, iterations=5, epsilon=5.0, mode='sum', theta=0.9):
        def choose_rank(singularValues, mode='sum', theta=0.9):
            temp = n = 0
            if mode == 'sum':
                sum_s = sum(singularValues)
                for i in singularValues:
                    temp += i
                    n += 1
                    if temp > sum_s * theta:
                        break
            elif mode == 'square_sum':
                temp = n = 0
                sum_s = sum(singularValues ** 2)
                for i in singularValues:
                    temp += i ** 2
                    n += 1
                    if temp > sum_s * theta:
                        break
            return n
        S = S0.copy()
        L, s, Rt = np.linalg.svd(S, full_matrices=False)
        # Step 2: Choose rank n
        n = choose_rank(s, mode=mode, theta=theta)
        print('rank ', n)
        matrix_dim = S.shape[0]


        n_users = len(user_data_list)
        batch_size = n_users // iterations

        for i in range(iterations):
            
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size if i < iterations - 1 else n_users
            current_user_data = user_data_list[start_idx:end_idx]

            L, s, Rt = np.linalg.svd(S, full_matrices=False)

            Lr = np.mat(L[:, :n])
            Rr = np.mat(Rt[:n, :])
            
            # Streaming process: accumulate perturbed vectors
            sum_vector = np.zeros(n * n)
            count = 0
            
            for user_items in current_user_data:
                user_items_set = set(user_items)
                selected_values = [1 if (idx <= domain_size and idx in user_items_set) else 0 for idx in sorted_indices]
                selected_values = np.array(selected_values)
                
                sub_matrix = selected_values.reshape((d, d))

                user_proj = (Lr.T @ np.mat(sub_matrix)) @ Rr.T
                flat_vector = np.asarray(user_proj).flatten()

                perturbed_vec = self.perturb_vector(flat_vector, r=1, alpha=epsilon)

                sum_vector += perturbed_vec
                count += 1
            
            if count > 0:
                avg_vector = sum_vector / count

                S_new = avg_vector.reshape(n, n)
                S = Lr @ np.mat(S_new) @ Rr
                print('SSSS', S.shape)
            else:
                print("Warning: No users in batch")

        return np.asarray(S)


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

    def remove_non_top_items(self,data, top_d_list):

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

        return data, data.user_use



    def recover_items_from_top_list(self,mapped_list, top_d_list):

        reverse_mapping = {new_id: original_id for new_id, original_id in enumerate(top_d_list)}

        return [reverse_mapping[i] for i in mapped_list if i in reverse_mapping]
    def find_top_k(self, data,ratios, epsilon, top_d, top_k, iterations,b,m):

        metrics_itemd = MetricsItem(data, top_k=top_d)
        real_topd_item = metrics_itemd.true_item_dict

        metrics_itemk = MetricsItem(data, top_k=top_k)
        real_topk_item = metrics_itemk.true_item_dict

        group1, group2, group3 = self.split_data_by_ratio(data, ratios)

        print('==================================================================')

        s0,k_list,bit_list = self.init_matrix(group1,epsilon,topk=m,b=b)

        S_iter = self.svd_iteration(s0, group2.data, k_list, d=m, domain_size=group2.dict_size, iterations=iterations, epsilon=epsilon,theta=0.9)
        top_d_list = self.get_topd_original_indices(S_iter,k_list,top_d)


        group3,user_num = self.remove_non_top_items(group3, top_d_list)

        svim = SVIM(group3, top_k=top_k, epsilon=epsilon)
        estimate_list_svim, estimate_freq_svim = svim.find_singleton_origin(user_num)

        estimate_list_svim = self.recover_items_from_top_list(estimate_list_svim,top_d_list)

        estimate_dict_svim = svim.convert_result2dict(estimate_list_svim, estimate_freq_svim)
        f1 = metrics_itemk.item_f1(estimate_dict_svim)
        ncr = metrics_itemk.item_ncr(estimate_dict_svim)
        mcr = metrics_itemk.cal_mcr(estimate_dict_svim)
        return f1, ncr, mcr


eps_list = [1,2,3,4]
k_list = [16]

# R1: 5% 10% 15% 20% 25%
kos_r1_list = [46,65,79,91,102]
mvl_r1_list = [103, 145, 178, 205, 229]
ub_r1_list = [163, 230, 282, 325, 363]
syn65_r1_list=[58,81,100,115,128]
syn131_r1_list=[81,115,140,162,181]

# R2
r2_list = [0.1,0.2,0.4,0.6,0.8]

iter_list = [5]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../results/bss-kos-{timestamp}.txt")
ratio_list = [
    [0.10, 0.10, 0.80]
]

for ratio in ratio_list:
    for k in k_list:
        for eps in eps_list:
            for m in kos_r1_list:
                for nn in r2_list:
                    n = np.ceil(nn*m*m).astype(int)
                    for iters in iter_list:
                        f1_list=[]
                        ncr_list=[]
                        mcr_list=[]
                        for _ in range(3):
                            # data = Data(dataname='syn65k', limit=100000, domain_size=65536, user_total=100000)
                            # data = Data(dataname='syn130k', limit=100000, domain_size=131073, user_total=100000)
                            # data = Data(dataname='movielens', limit=100000, domain_size=209163, user_total=162414)
                            data = Data(dataname='kosarak', limit=100000, domain_size=41270, user_total=990002)
                            # data = Data(dataname='UserBehavior1300', limit=100000, domain_size=528150, user_total=863327)
                            bss = BSSHHI(data)
                            f1, ncr, mcr = bss.find_top_k(data,ratios=ratio, epsilon=eps, top_d=n, top_k=k, iterations=iters, b=16,m=m)
                            with open(filename,"a") as f:
                                print(f"[eps={eps},ratios={ratio},k={k},x1={m}*{m}={m*m},x2={n}] f1={f1},ncr={ncr}", file=f)
                            f1_list.append(f1)
                            ncr_list.append(ncr)
                            mcr_list.append(mcr)
                        f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
                        ncr_mean, ncr_std = np.mean(ncr_list), np.std(ncr_list)
                        mcr_mean, mcr_std = np.mean(mcr_list), np.std(mcr_list)
                        with open(filename, "a") as f:
                            print(
                                f"[ratio = {ratio}, eps={eps},k={k},x1={m}*{m}={m * m},x2={nn}] f1={f1_mean:.4f}+-"
                                f"{f1_std:.4f},ncr={ncr_mean:.4f}+-{ncr_std:.4f}, mcr={mcr_mean:.4f}+-{mcr_std:.4f}", file=f)

