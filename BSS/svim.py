import json
import math
import random
import time

import numpy as np
import scipy.stats as st
import sys
from collections import defaultdict
from datetime import datetime
sys.path.append("..")
import fo, mechnism
from data_rm import Data
from metrics_item import MetricsItem
from typing import List, Tuple

class SVIM:
    def __init__(self, data, top_k=50, epsilon=2.0):
        self.data = data
        self.top_k = top_k
        self.epsilon = epsilon
        print(data.dict_size)

    

    def find_singleton_origin(self, single_test_user):
        # for BSSHHI and SVIM
        phase1_user = int(single_test_user * 0.4)  # 40%
        phase2_user = phase1_user + int(0.1 * single_test_user)  # 50%
        phase3_user = int(single_test_user)  # 100% 
        true_singleton_dist = self.data.test_single(0, phase1_user)
        est_singleton_dist = fo.lh(true_singleton_dist, self.epsilon)
        top_singleton = 2 * self.top_k
        singleton_list, value_result = self.build_result(est_singleton_dist, range(len(est_singleton_dist)),
                                                         top_singleton)

        key_result = {}
        for i in range(len(singleton_list)):
            key_result[(singleton_list[i],)] = i
        length_percentile = 0.9
        length_distribution = self.test_length_singleton(phase1_user + 1, phase2_user, len(singleton_list),
                                                         singleton_list)
        length_limit = self.find_percentile_set(length_distribution, length_percentile, phase2_user - phase1_user)
        # length_limit = 16
        use_grr, eps = self.set_grr(key_result, length_limit)
        true_singleton_dist = self.data.test_singleton_cand_limit(phase2_user + 1, phase3_user, key_result,
                                                                  set(singleton_list), length_limit)
        print('true_singleton_dist',true_singleton_dist)
        if use_grr:
            value_estimates = fo.rr(true_singleton_dist, eps)[:-1]
        else:
            value_estimates = fo.lh(true_singleton_dist, eps)[:-1]
        self.update_tail_with_reporting_set(length_limit, length_distribution, value_estimates)

        value_estimates *= single_test_user * length_limit / (phase3_user - phase2_user)
        top_singleton = self.top_k
        key_list, est_freq = self.build_result(value_estimates, singleton_list, top_singleton)

        print('***********SVIM Result**********')
        true_item_cand = self.data.true_item_cand(self.top_k)
        key_list_true = []
        for key in true_item_cand.keys():
            key_list_true.append(key[0])
        print('True top singleton of all users??:', true_item_cand)
        print('Top %d estimate singleton reported after step3(represent all users):' % self.top_k)
        for i in range(len(key_list)):
            print('{}:{}  '.format(key_list[i], int(np.nan_to_num(est_freq[i]))), end=' ')
        print('\nThe right numbers of identified items:', len(set(key_list_true).intersection(set(key_list))))
        return key_list, est_freq

    def find_singleton(self, single_test_user):
        # for SVD-FO
        phase1_user = int(single_test_user * 0.4)  # 40%
        phase2_user = phase1_user + int(0.1 * single_test_user)  # 50%
        phase3_user = int(single_test_user * 0.9)  # 100% 
        phase4_user = single_test_user
        true_singleton_dist = self.data.test_single(0, phase1_user)
        est_singleton_dist = fo.lh(true_singleton_dist, self.epsilon)
        top_singleton = 2 * self.top_k
        singleton_list, value_result = self.build_result(est_singleton_dist, range(len(est_singleton_dist)),
                                                         top_singleton)

        key_result = {}
        for i in range(len(singleton_list)):
            key_result[(singleton_list[i],)] = i
        length_percentile = 0.9
        length_distribution = self.test_length_singleton(phase1_user + 1, phase2_user, len(singleton_list),
                                                         singleton_list)
        length_limit = self.find_percentile_set(length_distribution, length_percentile, phase2_user - phase1_user)
        # length_limit = 16
        use_grr, eps = self.set_grr(key_result, length_limit)
        true_singleton_dist = self.data.test_singleton_cand_limit(phase2_user + 1, phase3_user, key_result,
                                                                  set(singleton_list), length_limit)
        print('true_singleton_dist',true_singleton_dist)
        if use_grr:
            value_estimates = fo.rr(true_singleton_dist, eps)[:-1]
        else:
            value_estimates = fo.lh(true_singleton_dist, eps)[:-1]
        self.update_tail_with_reporting_set(length_limit, length_distribution, value_estimates)
        value_estimates *= single_test_user * length_limit / (phase3_user - phase2_user)
        top_singleton = self.top_k
        key_list, est_freq = self.build_result(value_estimates, singleton_list, top_singleton)

        print('***********SVIM Result**********')
        true_item_cand = self.data.true_item_cand(self.top_k)
        key_list_true = []
        for key in true_item_cand.keys():
            key_list_true.append(key[0])
        print('True top singleton of all users??:', true_item_cand)
        print('Top %d estimate singleton reported after step3(represent all users):' % self.top_k)
        for i in range(len(key_list)):
            print('{}:{}  '.format(key_list[i], int(np.nan_to_num(est_freq[i]))), end=' ')
        print('\nThe right numbers of identified items:', len(set(key_list_true).intersection(set(key_list))))
        print()
        length_distribution1 = self.test_length_singleton(phase3_user + 1, phase4_user, len(key_list), key_list)
        length_limit1 = self.find_mean_lenghth(length_distribution1, phase4_user - phase3_user)
        return key_list, est_freq, length_limit1

    # ===== auxiliary functions for singletons
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

    def find_percentile_set(self, length_distribution, length_percentile, user, private=True):
        current_total = 0
        if private:
            threshold = st.norm.ppf(1 - 0.05 / 2 / self.top_k) * math.sqrt(
                4 * math.e ** self.epsilon / (math.e ** self.epsilon - 1) ** 2 * user)
            for i in range(len(length_distribution)):
                if length_distribution[i] < threshold:
                    length_distribution[i] = 0
        total = sum(length_distribution)
        i = 1
        for i in range(1, len(length_distribution)):
            current_total += length_distribution[i]
            if total != 0.0:
                if current_total / total > length_percentile:
                    break
        return i

    def find_mean_lenghth(self, length_distribution, user, private=True):
        if private:
            threshold = st.norm.ppf(1 - 0.05 / 2 / self.top_k) * math.sqrt(
                4 * math.e ** self.epsilon / (math.e ** self.epsilon - 1) ** 2 * user)
            for i in range(len(length_distribution)):
                if length_distribution[i] < threshold:
                    length_distribution[i] = 0
        total = sum(length_distribution)
        suml = 0
        for i in range(1, len(length_distribution)):
            suml += length_distribution[i] * i

        if total != 0.0:
            return round(suml / total)
        else:
            return 1

    def set_grr(self, new_cand_dict, length_limit):  # length_limit = l
        eps = self.epsilon
        use_grr = False
        if len(new_cand_dict) < length_limit * math.exp(self.epsilon) * (4 * length_limit - 1) + 1:
            eps = math.log(length_limit * (math.exp(self.epsilon) - 1) + 1)
            use_grr = True
        return use_grr, eps

    def test_length_singleton(self, user_start, user_end, length_limit, cand_dict):
        true_length_dist = self.data.test_length_cand(user_start, user_end, cand_dict, length_limit)
        est_length_dist = fo.lh(true_length_dist, self.epsilon)
        return est_length_dist

    def update_tail_with_reporting_set(self, length_limit, length_distribution_set, value_result):
        addi_total_item = 0
        for i in range(length_limit + 1, len(length_distribution_set)):
            addi_total_item += length_distribution_set[i] * (i - length_limit)
            if length_distribution_set[i] == 0:
                break
        total_item = sum(value_result)
        if total_item:
            ratio = addi_total_item * 1.0 / total_item
        else:
            ratio = 0
        for i in range(len(value_result)):
            value_result[i] *= (1.0 + ratio)
        return value_result

    def update_tail_with_reporting_set_1(self, length_limit, length_distribution_set, value_result):
        addi_total_item = 0
        for i in range(length_limit + 1, len(length_distribution_set)):
            addi_total_item += length_distribution_set[i] * (i - length_limit)
            if length_distribution_set[i] == 0:
                break
        total_item = sum(length_distribution_set)
        ratio = total_item / (total_item - addi_total_item)

        for i in range(len(value_result)):
            value_result[i] *= ratio
        return value_result

    def convert_result2dict(self, key_list, est_freq):
        estimated_item_dict = {}
        for i in range(len(key_list)):
            estimated_item_dict[tuple([key_list[i]])] = int(est_freq[i])
        return estimated_item_dict

    # gt = ground truth
    def find_singleton_gt(self, single_test_user):
        phase1_user = int(single_test_user * 0.4)  # 40%
        phase2_user = phase1_user + int(0.1 * single_test_user)  # 50%
        phase3_user = int(single_test_user * 0.9)  # 100%
        phase4_user = single_test_user
        true_singleton_dist = self.data.test_single(0, phase1_user)
        est_singleton_dist = true_singleton_dist.astype(np.float64)
        top_singleton = 4 * self.top_k
        singleton_list, value_result = self.build_result(est_singleton_dist, range(len(est_singleton_dist)),
                                                         top_singleton)
        key_result = {}
        for i in range(len(singleton_list)):
            key_result[(singleton_list[i],)] = i
        length_percentile = 0.9

        length_distribution = self.data.test_length_cand(phase1_user + 1, phase2_user, singleton_list, len(singleton_list))
        length_limit = self.find_percentile_set(length_distribution, length_percentile, phase2_user - phase1_user,private=False)

        true_singleton_dist = self.data.test_singleton_cand_limit(phase2_user + 1, phase3_user, key_result,
                                                                  set(singleton_list), length_limit)
        value_estimates = true_singleton_dist[:-1].astype(np.float64)
        self.update_tail_with_reporting_set(length_limit, length_distribution, value_estimates)
        value_estimates *= single_test_user * length_limit / (phase3_user - phase2_user) / 0.3
        top_singleton = self.top_k
        key_list, est_freq = self.build_result(value_estimates, singleton_list, top_singleton)

        print('***********SVIM Result**********')
        true_item_cand = self.data.true_item_cand(self.top_k)
        key_list_true = []
        for key in true_item_cand.keys():
            key_list_true.append(key[0])
        print('True top singleton of all users:', true_item_cand)
        print('Top %d estimate singleton reported after step3(represent all users):' % self.top_k)
        for i in range(len(key_list)):
            print('{}:{}  '.format(key_list[i], int(np.nan_to_num(est_freq[i]))), end=' ')
        print('\nThe right numbers of identified items:', len(set(key_list_true).intersection(set(key_list))))
        print()
        # estimate the length of users data intersected with topk items in order to clip domain of message(reported singular values)
        length_distribution1 = self.data.test_length_cand(phase3_user + 1, phase4_user, key_list, len(key_list))
        length_limit1 = self.find_mean_lenghth(length_distribution1, phase4_user - phase3_user,private=False)
        return key_list, est_freq, length_limit1




if __name__ == '__main__':
    eps_list = [1,1.5,2,2.5,3,3.5,4]
    k_list = [80,96,112,128]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/svim_kos_k80-128_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump([], f)
    for top_kk in k_list:
        for eps in eps_list:
            elapsed_time_mean = 0
            for  repeat in range(10):  
                t0 = time.perf_counter()   
                f1_list = []
                ncr_list = []
                mcr_list = []
                data_original = Data(dataname='kosarak', limit=100000, domain_size=41270, user_total=990002)
                # data_original = Data(dataname='syn130k', limit=100000, domain_size=131073, user_total=100000)
                # data_original = Data(dataname='syn65k', limit=100000, domain_size=65536, user_total=100000)
                # data_original = Data(dataname='UserBehavior1300', limit=100000, domain_size=528150, user_total=863327)
                # data_original = Data(dataname='movielens', limit=100000, domain_size=209163, user_total=162414)
                metrics_item = MetricsItem(data_original, top_k=top_kk)
            
                svim = SVIM(data_original, top_k=top_kk, epsilon=eps)
                estimate_list_svim, estimate_freq_svim = svim.find_singleton_origin(100000)
                estimate_dict_svim = svim.convert_result2dict(estimate_list_svim, estimate_freq_svim)
                f1 = metrics_item.item_f1(estimate_dict_svim)
                ncr = metrics_item.item_ncr(estimate_dict_svim)
                mcr = metrics_item.cal_mcr(estimate_dict_svim)
                elapsed_time = time.perf_counter() - t0 
                elapsed_time_mean=elapsed_time_mean+elapsed_time
                print(f"[eps={eps}, k={top_kk}] → F1: {f1:.4f}, NCR: {ncr:.4f}, MCR: {mcr:.4f}")
                f1_list.append(f1)
                ncr_list.append(ncr)
                mcr_list.append(mcr)
                single_result = {
                    'epsilon': eps,
                    'top_k': top_kk,
                    'repeat_id': repeat + 1,
                    'f1': round(f1, 4),
                    'ncr': round(ncr, 4),
                    'mcr': round(mcr ,4),
                    'elapsed_time': elapsed_time,
                }
                with open(filename, "r+") as f:
                    data = json.load(f)
                    data.append(single_result)
                    f.seek(0)
                    json.dump(data, f, indent=2)

            f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
            ncr_mean, ncr_std = np.mean(ncr_list), np.std(ncr_list)
            mcr_mean, mcr_std = np.mean(mcr_list), np.std(mcr_list)

            print(f"[eps={eps}, k={top_kk}] → F1: {f1_mean:.4f} ± {f1_std:.4f}, NCR: {ncr_mean:.4f}, MCR: {mcr_mean:.4f} ± {mcr_std:.4f}")
            result_entry = {
                "epsilon": eps,
                "top_k": top_kk,
                "f1_mean": round(f1_mean, 4),
                "f1_std": round(f1_std, 4),
                "ncr_mean": round(ncr_mean, 4),
                "ncr_std": round(ncr_std, 4),  
                "mcr_mean": round(mcr_mean, 4),
                "mcr_std": round(mcr_std, 4),
                "elapsed_time_mean": elapsed_time_mean/10,  
            }
            with open(filename, "r+") as f:
                data = json.load(f)
                data.append(result_entry)
                f.seek(0)
                json.dump(data, f, indent=2)

    print(f"\nAll results saved to: {filename}")
