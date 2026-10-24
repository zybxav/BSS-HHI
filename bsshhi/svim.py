# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy.stats as st
from . import frequency_oracles as fo

class SVIM:

    def __init__(self, data, top_k=50, epsilon=2.0):
        self.data = data
        self.top_k = top_k
        self.epsilon = epsilon

    def find_singleton_origin(self, single_test_user):
        phase1_user = int(single_test_user * 0.4)
        phase2_user = phase1_user + int(0.1 * single_test_user)
        phase3_user = int(single_test_user)
        true_singleton_dist = self.data.test_single(0, phase1_user)
        est_singleton_dist = fo.lh(true_singleton_dist, self.epsilon)
        top_singleton = 2 * self.top_k
        singleton_list, value_result = self.build_result(est_singleton_dist, range(len(est_singleton_dist)), top_singleton)
        key_result = {}
        for i in range(len(singleton_list)):
            key_result[singleton_list[i],] = i
        length_percentile = 0.9
        length_distribution = self.test_length_singleton(phase1_user + 1, phase2_user, len(singleton_list), singleton_list)
        length_limit = self.find_percentile_set(length_distribution, length_percentile, phase2_user - phase1_user)
        use_grr, eps = self.set_grr(key_result, length_limit)
        true_singleton_dist = self.data.test_singleton_cand_limit(phase2_user + 1, phase3_user, key_result, set(singleton_list), length_limit)
        if use_grr:
            value_estimates = fo.rr(true_singleton_dist, eps)[:-1]
        else:
            value_estimates = fo.lh(true_singleton_dist, eps)[:-1]
        self.update_tail_with_reporting_set(length_limit, length_distribution, value_estimates)
        value_estimates *= single_test_user * length_limit / (phase3_user - phase2_user)
        top_singleton = self.top_k
        key_list, est_freq = self.build_result(value_estimates, singleton_list, top_singleton)
        return (key_list, est_freq)

    def build_result(self, value_estimates, key_list, top_singleton):
        sorted_indices = np.argsort(value_estimates)
        key_result = []
        value_result = []
        for j in sorted_indices[-top_singleton:]:
            key_result.append(key_list[j])
            value_result.append(value_estimates[j])
        key_result.reverse()
        value_result.reverse()
        return (key_result, value_result)

    def find_percentile_set(self, length_distribution, length_percentile, user, private=True):
        current_total = 0
        if private:
            threshold = st.norm.ppf(1 - 0.05 / 2 / self.top_k) * math.sqrt(4 * math.e ** self.epsilon / (math.e ** self.epsilon - 1) ** 2 * user)
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

    def set_grr(self, new_cand_dict, length_limit):
        eps = self.epsilon
        use_grr = False
        if len(new_cand_dict) < length_limit * math.exp(self.epsilon) * (4 * length_limit - 1) + 1:
            eps = math.log(length_limit * (math.exp(self.epsilon) - 1) + 1)
            use_grr = True
        return (use_grr, eps)

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
            value_result[i] *= 1.0 + ratio
        return value_result

    def convert_result2dict(self, key_list, est_freq):
        estimated_item_dict = {}
        for i in range(len(key_list)):
            estimated_item_dict[tuple([key_list[i]])] = int(est_freq[i])
        return estimated_item_dict
