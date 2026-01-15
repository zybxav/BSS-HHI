import heapq
import random
from datetime import datetime
import numpy as np
import sys
sys.path.append("../..")
from svim import SVIM
from data_rm import Data
from metrics_item import MetricsItem
import fo
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
        print('encode length',len(result))
        return result

    def top_k_binary_numbers(self, bit_list, bit_freq, b, k):
        bit_prob_map = {bit: freq for bit, freq in zip(bit_list, bit_freq)}
        print('bit_prob_map', bit_prob_map)
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
                    prob *= (1 - freq)

            result.append((prob, binary))

        top_k = heapq.nlargest(k, result)

        binary_list = [binary for _, binary in top_k]
        freq_list = [prob for prob, _ in top_k]

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
        return k_bin_decimal


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
        assert abs(sum(ratios) - 1.0) < 1e-6, "sum of ratios must be 1"
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

    def build_submatrices_from_data(self, data_obj, sorted_indices, d):
        domain_size = data_obj.dict_size

        sub_matrices = []
        for user_items in data_obj.data:
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

        return sub_matrices,sorted_indices

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

    def svd_iteration(self, S0, user_group2, theta,iterations=5, epsilon=5.0, mode='sum'):
        def choose_rank(singularValues,theta, mode='sum'):
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
        n = choose_rank(s, mode=mode, theta=theta)
        print('rank ', n)
        matrix_dim = S.shape[0]

        user_groups = np.array_split(user_group2, iterations)

        for i in range(iterations):
            L, s, Rt = np.linalg.svd(S, full_matrices=False)

            Lr = np.mat(L[:, :n])  
            Rr = np.mat(Rt[:n, :])  
        
            current_user_group = user_groups[i]  

            projected_user_matrices = []
            for user_mat in current_user_group:
                user_proj = (Lr.T @ np.mat(user_mat)) @ Rr.T
                projected_user_matrices.append(np.asarray(user_proj))


            flat_vectors = [mat.flatten() for mat in projected_user_matrices]
            flat_matrix = np.vstack(flat_vectors)


            avg_vector = self.l2_ball_mechanism(r=1, alpha=epsilon, vectors=flat_matrix)
            avg_vector = np.mean(avg_vector, axis=0)

            S = avg_vector.reshape(n, n)
            S = Lr @ np.mat(S) @ Rr
            print('S', S.shape)

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

    def build_result(self, value_estimates, key_list, top_singleton):
        sorted_indices = np.argsort(value_estimates)
        key_result = []
        value_result = []
        for j in sorted_indices[-top_singleton:]:
            key_result.append(key_list[j])
            value_result.append(value_estimates[j])
        key_result.reverse()
        value_result.reverse()
        return key_result, value_result

    def find_top_k(self, data, epsilon, top_d, top_k,ratios, iterations,b,m,theta):
        metrics_itemk = MetricsItem(data, top_k=top_k)
        real_topk_item = metrics_itemk.true_item_dict
        print('k',real_topk_item)
        group1, group2= self.split_data_by_ratio(data, ratios)

        print('==================================================================')

        k_list= self.init_matrix(group1,epsilon,topk=m,b=b)
        k_dict = {(k_list[i],): i for i in range(len(k_list))}

        true_singleton_list = group2.test_singleton_cand_limit(0,len(group2.data),k_dict,set(k_list),1)
        est_dict = fo.lh(true_singleton_list,epsilon)[:-1]
        print('k_list',k_list)
        print('k_dict', k_dict)

        key_result, value_result = self.build_result(est_dict, k_list, top_k)
        est = {
            (k if isinstance(k, tuple) else (k,)): v
            for k, v in zip(key_result, value_result)
        }
        print('est', est)

        f1 = metrics_itemk.item_f1(est)
        ncr = metrics_itemk.item_ncr(est)
        mcr = metrics_itemk.cal_mcr(est)
        return f1, ncr, mcr


eps_list = [4]
k_list = [32]
ratios_list = [[0.2,0.8]]

kos_m_param_list = [46,65,79,91,102]
mvl_m_param_list = [103, 145, 178, 205, 229]
ub_m_param_list = [163, 230, 282, 325, 363]
syn65_m_param_list=[58,81,100,115,128]
syn131_m_param_list=[81,115,140,162,181]

kos_n_param_list = [413, 826, 1239, 1652, 2065]
mvl_n_param_list = [2092, 4184, 6275, 8367, 10459]
ub_n_param_list = [5281, 10563, 15844, 21126, 26407]
syn65_n_param_list = [655,1311,1966,2621,3277]
syn131_n_param_list = [1311,2621,3932,5242,6553]

kos_m_list = [46]
mvl_m_list = [205]
syn65_m_list=[128]
syn131_m_list=[115]
ub_m_list = [230]

kos_n_list = [413]
mvl_n_list = [4203]
syn65_n_list = [1639]
syn131_n_list = [2621]
ub_n_list = [5281]


iter_list = [5]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"../BSSFO/BSS-FO-65k-set-k32e4-{timestamp}.txt" #change with dataset
for ratios in ratios_list:
    for k in k_list:
        for eps in eps_list:
            for m in syn65_m_list: #change with dataset
                for n in syn65_n_list: # no need in BSS-FO
                    for iters in iter_list:
                        f1_list=[]
                        ncr_list=[]
                        mcr_list = []
                        for _ in range(6):
                            print(ratios)
                            # data = Data(dataname='movielens', limit=100000, domain_size=209163, user_total=162414)
                            # data = Data(dataname='kosarak', limit=100000, domain_size=41270, user_total=990002)
                            data = Data(dataname='syn65k', limit=100000, domain_size=65536, user_total=100000)
                            #data = Data(dataname='Zipf-131k', limit=100000, domain_size=131072, user_total=100000)
                            # data = Data(dataname='UserBehavior', limit=100000, domain_size=528150, user_total=863327)
                            bssfo = BSSHHI(data)
                            # Kosarak: b=16 Movielens: b=18 Userbeh: b=20 Zipf-65k: b=17 Zipf-131k: b=18
                            # theta=0.9 or 1.0 following the results in the paper.
                            f1, ncr, mcr = bssfo.find_top_k(data, epsilon=eps, top_d=n, top_k=k,ratios=ratios, iterations=iters, b=17,m=m,theta=0.9)
                            with open(filename,"a") as f:
                                print(f"[ratios={ratios},eps={eps},k={k},x1={m}*{m}={m*m}] f1={f1},ncr={ncr},mcr={mcr}", file=f)
                            f1_list.append(f1)
                            ncr_list.append(ncr)
                            mcr_list.append(mcr)
                        f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
                        ncr_mean, ncr_std = np.mean(ncr_list), np.std(ncr_list)
                        mcr_mean, mcr_std = np.mean(mcr_list), np.std(mcr_list)
                        with open(filename, "a") as f:
                            print(f"[eps={eps},k={k},x1={m}*{m}={m * m}] f1={f1_mean:.4f}+-"
                                  f"{f1_std:.4f},ncr={ncr_mean:.4f}+-{ncr_std:.4f},mcr={mcr_mean:.4f}+-{mcr_std:.4f}", file=f)