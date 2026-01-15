import heapq
import itertools
import math
import random
import pprint
import numpy as np
import scipy.stats as st
from sklearn import preprocessing
from numpy.linalg import svd
import sys
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, linewidth=200, suppress=True)
sys.path.append("..")
import fo, mechnism, mechnism_hybird
from svim import SVIM
from svsm_rm import SVSM
from data_rm import Data
# from metrics import Metrics
from metrics_item import MetricsItem
# from svme import SVME


class LDP_RM:
    def __init__(self, data, epsilon, top_k, top_ks, top_kc, submat=0):
        self.data = data
        self.multi_test_user = len(self.data.data)
        self.epsilon = epsilon
        self.top_k = int(top_k)
        self.top_ks = int(top_ks)
        self.top_kc = int(top_kc)
        self.submat = submat
        if submat:
            subnum = (submat ** 2 + submat) // 2
            self.sub_top_ks = int(top_ks // subnum)
        else:
            self.sub_top_ks = int(top_ks)
        self.svsm = SVSM(data, epsilon=epsilon, top_k=top_k, top_ks=top_ks, top_kc=top_kc)

    def find_itemset_svd(self, task='ARM', method='AMN', singnum=0, use_group=True, group_num=1, test='test_constant'):

        print('----------------------LDP-RM Result----------------------')
        multi_test_user = self.multi_test_user
        single_test_user_percentile = 0.5
        single_test_user = int(single_test_user_percentile * multi_test_user)  # 30%
        itemset_test_user_percentile = 0.99
        svd_test_user = int(itemset_test_user_percentile * multi_test_user)  # 35%
        if method == 'gt':
            singletons, singleton_freq, length_limit = self.svsm.svim.find_singleton_gt(single_test_user)
        else:
            singletons, singleton_freq, length_limit = self.svsm.svim.find_singleton(single_test_user)

        set_cand_dict, hitrate = dict(), 0
        # need 3 steps
        if self.submat == 0:
            set_cand_dict, hitrate = self.find_itemset_svd_origin(singletons, singletons, singleton_freq, singleton_freq, single_test_user,
                                                                  svd_test_user, length_limit, sing_num=singnum, method=method, use_group=use_group,
                                                                  group_num=group_num, use_min=False, test=test, task=task)
        else:
            set_cand_dict, hitrate = self.find_itemset_svd_submat(singletons, singleton_freq, single_test_user, svd_test_user,
                                                                  length_limit, sing_num=singnum, method=method, use_group=use_group,
                                                                  group_num=group_num, use_min=False, test=test, task=task)

        return set_cand_dict,hitrate

    def find_item_svd(self, use_group=True, method='AMN', singnum=0, group_num=7, test='test_constant'):


        result_fre_dict,_ = self.find_itemset_svd(task='FISM', use_group=use_group, method=method, singnum=singnum,
                                                      group_num=group_num, test=test)
        item_dict = self.svdfo_build_results(result_fre_dict)
        return item_dict

    def find_itemset_svd_origin(self, singletons_row, singletons_col, singleton_freq_row, singleton_freq_col, single_test_user, svd_test_user,
                                length_limit, sing_num, method='AMN', use_min=False, use_group=True, group_num=1, test='', task='ARM'):

        print("task",task)
        multi_test_user = self.multi_test_user
        l = min(len(singletons_row), len(singletons_col))
        singletons_row, singletons_col = singletons_row[:l], singletons_col[:l]
        singleton_freq_row, singleton_freq_col = singleton_freq_row[:l], singleton_freq_col[:l]
        print('***********LDP-RM Result**********')

        # all combination hit rate
        all_c = list(itertools.product(singletons_row, singletons_col)) + list(itertools.product(singletons_col, singletons_row))
        for c in all_c:
            if c[0] == c[1]:
                all_c.remove(c)


        l = min(len(singletons_row), len(singletons_col))


        data = self.data.data[single_test_user:svd_test_user]

        init_matrix = self.build_init_matrix(singleton_freq_row, singleton_freq_col, multi_test_user, use_min)



        U, singularValues, V = svd(init_matrix, full_matrices=(singletons_row == singletons_col), hermitian=True)

        if sing_num and isinstance(sing_num,int):
            n = sing_num
        else:
            if sing_num==0:
                n = self.choose_rank(singularValues, mode='sum',theta=0.9)
            elif isinstance(sing_num,float):
                n = self.choose_rank(singularValues, mode='sum', theta=sing_num)

        matrix_aggregate = np.zeros([l, l])
        global vote_record
        global track

        if use_group:
            group_nums = group_num

            # shuffle each element symmetric
            if 'test' in test:
                user = (svd_test_user - single_test_user) // group_nums
                l = len(singleton_freq_row)
                # init_matrix
                matrix_init = np.zeros([l, l])
                lower = np.zeros([l, l])
                upper = np.zeros([l, l])


                _, matrix_init, lower, upper = self.generate_iterate_matrix(singleton_freq_row, singleton_freq_col, multi_test_user)

                matrix_iterate, shuffle_map = self.shuffle_matrix_all(matrix_init, singleton_freq_row, singleton_freq_col)

                i = 0
                vote_record = np.zeros([l, l, group_nums])
                level = np.zeros([l, l])
                level += 10
                level[np.diag_indices_from(level)] = 0
                while i < group_nums:
                    # shuffle lower and upper
                    lower_shuffle = np.zeros([l, l])
                    upper_shuffle = np.zeros([l, l])
                    if (singletons_row == singletons_col):
                        for j in range(l - 1):
                            for k in range(j + 1, l):
                                lower_shuffle[shuffle_map[(j, k)]] = lower[j, k]
                                upper_shuffle[shuffle_map[(j, k)]] = upper[j, k]
                        lower_shuffle += lower_shuffle.T
                        upper_shuffle += upper_shuffle.T
                    else:
                        for j in range(l):
                            for k in range(l):
                                lower_shuffle[shuffle_map[(j, k)]] = lower[j, k]
                                upper_shuffle[shuffle_map[(j, k)]] = upper[j, k]

                    matrix_iterate, level \
                        = self.find_true_itemsets_origin(matrix_iterate, level, shuffle_map, lower_shuffle, upper_shuffle, l,
                                                         data[user * i:user + user * i], length_limit, self.epsilon, singletons_row, singletons_col,
                                                         method, n)

                    # recover shuffle
                    recover_mat = np.zeros([l, l])
                    if (singletons_row == singletons_col):
                        for j in range(l - 1):
                            for k in range(j + 1, l):
                                recover_mat[j, k] = matrix_iterate[shuffle_map[(j, k)]]
                        recover_mat += recover_mat.T
                    else:
                        for j in range(l):
                            for k in range(l):
                                recover_mat[j, k] = matrix_iterate[shuffle_map[(j, k)]]

                    matrix_iterate = recover_mat.copy()
                    # print(matrix_iterate)

                    for j in range(l):
                        for k in range(l):
                            vote_record[j, k, i] = matrix_iterate[j, k]

                    matrix_iterate, shuffle_map = self.shuffle_matrix_all(matrix_init, singleton_freq_row, singleton_freq_col)
                    matrix_iterate = matrix_iterate.astype(np.float64)
                    i += 1

                track = np.zeros([l, l])
                for j in range(l):
                    for k in range(l):
                        for i in range(group_nums):
                            track[j, k] += vote_record[j, k, i] * (i + 1) / sum([ite for ite in range(group_nums + 1)])
                            # track[j,k] += vote_record[j, k, i] / group_nums
                # print(track)
                matrix_aggregate = track.copy()

        elif not use_group:
            print('this is not_iterative mode')
            matrix_aggregate = init_matrix.copy()

        matrix = matrix_aggregate.copy()
        # matrix = real_matrix
        ks = 'fre'
        if task == 'ARM' and self.submat:
            for i in range(l):
                matrix[i] = (matrix[i] / singleton_freq_row[i] * multi_test_user)
        if ks == 'f*c':
            fre_normalization = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            for i in range(l):
                matrix[i] = (matrix[i] / singleton_freq_row[i] * multi_test_user)
            con_normalization = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            fre_normalization[np.diag_indices_from(fre_normalization)] = 0
            con_normalization[np.diag_indices_from(con_normalization)] = 0
            for i in range(l):
                for j in range(l):
                    matrix[i, j] = fre_normalization[i, j] * con_normalization[i, j]

        # Now matrix is asymmetric matrix
        # find top-ks elements in the matrix; res:list
        res, dict_est = self.find_matrix_ks(matrix.copy(), singletons_row, singletons_col)
        # real_res, dict_real = self.find_matrix_ks(real_matrix.copy(), singletons)

        set_cand_dict = {}
        for j in range(len(res)):
            set_cand_dict[res[j]] = j
        set_cand_dict_copy = set(set_cand_dict.keys()).copy()
        for itemset in set_cand_dict_copy.copy():
            set_cand_dict_copy.add((itemset[1], itemset[0]))


        hit_rate = 0

        return set_cand_dict, hit_rate

    def find_itemset_svd_submat(self, singletons, singleton_freq, single_test_user, svd_test_user, length_limit,
                                sing_num, method='AMN', use_min=False, use_group=True, group_num=1, test='', task='ARM'):

        l = len(singletons)
        submat = self.submat
        sublen = l // submat
        sub_singletons = [singletons[sublen * i:sublen * (i + 1)] for i in range(submat)]
        sub_singleton_freq = [singleton_freq[sublen * i:sublen * (i + 1)] for i in range(submat)]

        subnum = (submat ** 2 + submat) // 2
        each_subnum = (svd_test_user - single_test_user) // subnum
        set_cand_dict_list = []
        hitrate_list = []
        m = 0
        for i in range(submat):
            for j in range(i, submat):
                print('___________ ', m + 1, ' sub matirx ___________')

                set_cand_dictm, hitratem = self.find_itemset_svd_origin(sub_singletons[i], sub_singletons[j], sub_singleton_freq[i],
                                                                        sub_singleton_freq[j], single_test_user + each_subnum * m,
                                                                        single_test_user + each_subnum * (m + 1), length_limit,
                                                                        sing_num, method=method, use_min=use_min, use_group=use_group,
                                                                        group_num=group_num, test=test, task=task)
                set_cand_dict_list.append(set_cand_dictm)
                hitrate_list.append(hitratem)
                m += 1

        count = 0
        set_cand_dict = {}
        for set in set_cand_dict_list:
            for itemset in set.keys():
                set_cand_dict[itemset] = count
                count += 1
        hitrate_sum = 0
        for hitrate in hitrate_list:
            hitrate_sum += hitrate
        # print(set_cand_dict)
        print('final hitrate', round(hitrate_sum, 2))
        print()

        return set_cand_dict, hitrate_sum

    def generate_rules(self, task, singletons, singleton_freq, single_test_user_percentile, svd_test_user, multi_test_user, set_cand_dict):
        # print('sing_freq',singleton_freq)
        percentile_test_user = svd_test_user + int(0.2 * (multi_test_user - svd_test_user))

        # step 4: itemset size distribution
        length_percentile = 0.9
        length_distribution_set = self.svsm.test_length_itemset(svd_test_user + 1, percentile_test_user, len(set_cand_dict), set_cand_dict,
                                                                self.epsilon)
        length_limit = self.svsm.svim.find_percentile_set(length_distribution_set, length_percentile, percentile_test_user - svd_test_user)

        # step 5: itemset est
        true_itemset_dist = self.data.test_itemsets_cand_limit(percentile_test_user + 1, multi_test_user, set_cand_dict, length_limit)


        use_grr, eps = self.svsm.svim.set_grr(true_itemset_dist, length_limit)

        if use_grr:
            set_freq = fo.rr(true_itemset_dist, eps)[:-1]
        else:
            set_freq = fo.lh(true_itemset_dist, eps)[:-1]

        set_freq *= multi_test_user * (1 - single_test_user_percentile) * length_limit / (
                multi_test_user - percentile_test_user)
        self.svsm.svim.update_tail_with_reporting_set(length_limit, length_distribution_set, set_freq)
        set_freq /= (1 - single_test_user_percentile)
        # print('set_freq',set_freq)

        result_fre_dict = {}
        result_conf_dict = {}
        if task == 'ARM':
            confidence_dict = {}
            res = list(dict(sorted(set_cand_dict.items(), key=lambda x: x[1])).keys())

            for i in range(len(set_freq)):
                a, b = singletons.index(res[i][0]), singletons.index(res[i][1])
                confidence_dict[tuple([res[i][0], res[i][1]])] = set_freq[i] / singleton_freq[a]
                confidence_dict[tuple([res[i][1], res[i][0]])] = set_freq[i] / singleton_freq[b]

            confidence_dict = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))

            itemset_list = list(confidence_dict.keys())[:self.top_kc]
            if len(itemset_list) < self.top_kc:
                print(len(itemset_list))
                raise IndexError('len(itemset_list)< self.top_kc')

            for i in range(int(len(itemset_list))):
                result_conf_dict[itemset_list[i]] = confidence_dict[itemset_list[i]]

            set_freq_2 = set_freq.copy()
            for key in result_conf_dict.keys():
                if key in res:
                    result_fre_dict[key] = int(set_freq_2[res.index(key)])
                    # print(set_freq_2[res.index(key)])
                else:
                    key_r = tuple([key[1], key[0]])
                    result_fre_dict[key] = int(set_freq_2[res.index(key_r)])

        elif task == 'FISM':
            res = list(set_cand_dict.keys())
            result_fre_dict = self.svsm.build_set_result(set_freq, res, self.top_kc)


        return result_fre_dict, result_conf_dict

    def generate_rules_gt(self, task, singletons, singleton_freq, single_test_user_percentile, svd_test_user, multi_test_user, set_cand_dict):

        print(task)
        percentile_test_user = svd_test_user + int(0.2 * (multi_test_user - svd_test_user))
        # step 4: itemset size distribution
        length_percentile = 0.9
        length_distribution_set = self.data.test_length_itemset(svd_test_user + 1, percentile_test_user, set_cand_dict, len(set_cand_dict))
        length_limit = self.svsm.svim.find_percentile_set(length_distribution_set, length_percentile, percentile_test_user - svd_test_user,
                                                          private=None)

        # step 5: itemset est
        true_itemset_dist = self.data.test_itemsets_cand_limit(percentile_test_user + 1, multi_test_user, set_cand_dict, length_limit)

        set_freq = true_itemset_dist[:-1].astype(np.float64)

        set_freq *= multi_test_user * (1 - single_test_user_percentile) * length_limit / (multi_test_user - percentile_test_user)
        self.svsm.svim.update_tail_with_reporting_set(length_limit, length_distribution_set, set_freq)
        set_freq /= (1 - single_test_user_percentile)

        result_sup_dict = {}
        result_conf_dict = {}
        if task == 'RM':
            confidence_dict = {}
            res = list(dict(sorted(set_cand_dict.items(), key=lambda x: x[1])).keys())

            for i in range(len(set_freq)):
                a, b = singletons.index(res[i][0]), singletons.index(res[i][1])
                confidence_dict[tuple([res[i][0], res[i][1]])] = set_freq[i] / singleton_freq[a]

            confidence_dict = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))

            kc_relation_list = list(confidence_dict.keys())[:self.top_kc]
            if len(kc_relation_list) < self.top_kc:
                print(len(kc_relation_list))
                raise IndexError('len(itemset_list)< self.top_kc')

            for i in range(int(len(kc_relation_list))):
                result_conf_dict[kc_relation_list[i]] = confidence_dict[kc_relation_list[i]]

            set_freq_2 = set_freq.copy()
            for key in result_conf_dict.keys():
                if key in res:
                    result_sup_dict[key] = int(set_freq_2[res.index(key)])
                    # print(set_freq_2[res.index(key)])
                else:
                    key_r = tuple([key[1], key[0]])
                    result_sup_dict[key] = int(set_freq_2[res.index(key_r)])

        elif task == 'FISM':
            res = list(set_cand_dict.keys())
            result_sup_dict = self.svsm.build_set_result(set_freq, res, self.top_kc)


        return result_sup_dict, result_conf_dict

    # =============svd part================#

    def find_matrix_ks(self, matrix, singletons_row, singletons_col):


        candidate = []
        index_res = []
        dict = {}
        l = len(singletons_row)
        if 1:
            while len(candidate) < self.sub_top_ks:
                for items in index_res:
                    matrix[items[0], items[1]] = matrix[items[1], items[0]] = 0
                matrix[matrix == 0] = -100
                max_value = matrix.max()
                if max_value == -100:
                    print(max_value)
                    print(len(candidate))
                    break
                max_index = np.argwhere(max_value - matrix < 1e-7)
                for candidate_index in max_index:
                    if len(candidate) >= self.sub_top_ks:
                        break
                    # if candidate_index[0] != candidate_index[1]:
                    if 1:
                        alpha, beta = singletons_row[candidate_index[0]], singletons_col[candidate_index[1]]
                        temp_tuple = (min(alpha, beta), max(alpha, beta))
                        # temp_tuple = (alpha, beta)
                        if temp_tuple not in candidate:
                            candidate.append(temp_tuple)
                            # print(candidate)
                            dict[temp_tuple] = matrix[candidate_index[0], candidate_index[1]]
                            index_res.append((candidate_index[0], candidate_index[1]))
                        matrix[candidate_index[0], candidate_index[1]] = matrix[candidate_index[1], candidate_index[0]] = 0
                if len(candidate) >= min(self.sub_top_ks, l * (l - 1) // 2):
                    break
        return candidate, dict

    def build_init_matrix(self, singleton_freq_row, singleton_freq_col, all_user, use_min=False):

        l = len(singleton_freq_row)
        support_row = np.zeros(l)
        support_col = np.zeros(l)
        for i in range(l):
            support_row[i] = (singleton_freq_row[i] / all_user)
            support_col[i] = (singleton_freq_col[i] / all_user)
        matrix = np.zeros([l, l])
        for i in range(l):
            for j in range(l):
                if use_min:
                    matrix[i][j] = min(support_row[i], support_col[j])
                else:
                    matrix[i][j] = support_row[i] * support_col[j]
            if singleton_freq_row == singleton_freq_col:
                matrix[np.diag_indices_from(matrix)] = 0
        return matrix

    def generate_iterate_matrix(self, singleton_freq_row, singleton_freq_col, all_user):

        l = len(singleton_freq_row)
        init_indep = self.build_init_matrix(singleton_freq_row, singleton_freq_col, all_user, use_min=False)
        init_dep = self.build_init_matrix(singleton_freq_row, singleton_freq_col, all_user, use_min=True)
        y = init_dep
        # x = init_indep
        # level_Unit = [1 - math.log(i) / math.log(20) for i in range(20, 0, -1)]
        level_Unit = [i / 20 for i in range(0, 21)]
        x = init_dep * level_Unit[5]
        x = init_indep

        z = np.zeros([l, l])


        mat = np.zeros([l, l])
        mat_lower_bound = np.zeros([l, l])
        mat_mid = np.zeros([l, l])
        mat_upper_bound = np.zeros([l, l])
        for i in range(l):
            for j in range(l):
                mat[i, j] = mat[j, i] = np.random.choice([x[i, j], y[i, j], z[i, j]], p=[1 / 3, 1 / 3, 1 / 3])
                # mat[i, j] = mat[j, i] = np.random.choice([y[i, j], z[i, j]], p=[1 / 2, 1 / 2])
                mat_mid[i, j] = mat_mid[j, i] = x[i, j]
                mat_lower_bound[i, j] = mat_lower_bound[j, i] = z[i, j]
                mat_upper_bound[i, j] = mat_upper_bound[j, i] = y[i, j]
        return mat, mat_mid, mat_lower_bound, mat_upper_bound

    def shuffle_matrix(self, matrix_init, singletons):

        l = len(singletons)
        singletons_shuffle = singletons.copy()
        random.shuffle(singletons_shuffle)

        shuffle_map = {}
        for i in range(l):
            shuffle_map[singletons.index(singletons_shuffle[i])] = i
        matrix_shuffle = np.zeros([l, l])

        for i in range(l):
            for j in range(l):
                matrix_shuffle[shuffle_map[i], shuffle_map[j]] = matrix_init[i, j]

        return matrix_shuffle, singletons_shuffle, shuffle_map

    def shuffle_matrix_all(self, matrix_init, singleton_freq_row, singleton_freq_col):

        l = len(singleton_freq_row)
        matrix_shuffle = np.zeros([l, l])
        shuffle_map = {}

        if (singleton_freq_row == singleton_freq_col):
            triu = []
            for i in range(0, l - 1):
                for j in range(i + 1, l):
                    triu.append((i, j))
            random.shuffle(triu)
            triu_shuffle = triu.copy()
            k = 0
            for i in range(0, l - 1):
                for j in range(i + 1, l):
                    shuffle_map[(i, j)] = triu_shuffle[k]
                    matrix_shuffle[triu_shuffle[k][0], triu_shuffle[k][1]] = matrix_init[i, j]
                    k += 1
            matrix_shuffle += matrix_shuffle.T
        else:
            triu = []
            for i in range(l):
                for j in range(l):
                    triu.append((i, j))
            random.shuffle(triu)
            triu_shuffle = triu.copy()
            k = 0
            for i in range(l):
                for j in range(l):
                    shuffle_map[(i, j)] = triu_shuffle[k]
                    matrix_shuffle[triu_shuffle[k][0], triu_shuffle[k][1]] = matrix_init[i, j]
                    k += 1
        return matrix_shuffle, shuffle_map

    def recover_matrix(self, matrix_shuffle, shuffle_map):

        l = len(matrix_shuffle)
        recover_mat = np.zeros([l, l])
        for i in range(l):
            for j in range(l):
                recover_mat[i, j] = matrix_shuffle[shuffle_map[i], shuffle_map[j]]
        return recover_mat

    def choose_rank1(self, singularValues, mode='sum'):

        temp = n = 0
        if mode == 'sum':
            sum_s = sum(singularValues)
            for i in singularValues:
                temp += i
                n += 1
                if temp > sum_s * 0.9:  # threshold = 0.9
                    break
        elif mode == 'square_sum':
            temp = n = 0
            sum_s = sum(singularValues ** 2)
            for i in singularValues:
                temp += i ** 2
                n += 1
                if temp > sum_s * 0.9:  # threshold = 0.9
                    break
        return n

    def choose_rank(self, singularValues, mode='sum',theta=0.9):
        temp = n = 0
        if mode == 'sum':
            sum_s = sum(singularValues)
            for i in singularValues:
                temp += i
                n += 1
                if temp > sum_s * theta:  # threshold = 0.9
                    break
        elif mode == 'square_sum':
            temp = n = 0
            sum_s = sum(singularValues ** 2)
            for i in singularValues:
                temp += i ** 2
                n += 1
                if temp > sum_s * theta:  # threshold = 0.9
                    break
        return n


    def find_true_itemsets_origin(self, matrix, level, shuffle_map, lower, upper, l, data, length_limit, epsilon, singletons_row, singletons_col,
                                  method, n):

        U, singularValues, V = np.linalg.svd(matrix, hermitian=(singletons_row == singletons_col), full_matrices=True)

        Ur = np.mat(U[:, :n])
        Vr = np.mat(V[:n])
        clip_coefficient = 1.0
        domain_report = self.domain2(Ur, Vr, length_limit, clip_coefficient)
        recover_mat = Ur @ np.diag(singularValues[:n]) @ Vr
        data_temp = self.build_user_matrix(data, singletons_row, singletons_col, l)
        data_sf_list = []
        if (singletons_row == singletons_col):
            for d in data_temp:
                data_shuffle = np.zeros([l, l])
                for j in range(l - 1):
                    for k in range(j + 1, l):
                        data_shuffle[shuffle_map[(j, k)]] = d[j, k]
                data_shuffle += data_shuffle.T
                data_sf_list.append(data_shuffle)
        else:
            for d in data_temp:
                data_shuffle = np.zeros([l, l])
                for j in range(l):
                    for k in range(l):
                        data_shuffle[shuffle_map[(j, k)]] = d[j, k]
                data_sf_list.append(data_shuffle)


        report = self.report_diag(Ur, Vr, data_sf_list)

        perturb = getattr(mechnism, method, None)
        report_perturb = perturb(report, epsilon, domain_report)
        rp = report_perturb.copy()
        rp = rp.astype(np.float32)
        res = self.recover_frequent_itemset(report_perturb, Ur, Vr)


        r = res.copy()
        r = r.astype(np.float32)
        test_norm = 4
        tmp = 0
        if test_norm == 1:
            tmp = r - matrix
        elif test_norm == 2:
            matrix_normalization = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            r_normalization = (r - np.min(r)) / (np.max(r) - np.min(r))
            tmp = r_normalization - matrix_normalization
        elif test_norm == 3:
            matrix_std = preprocessing.StandardScaler().fit_transform(matrix)
            r_std = preprocessing.StandardScaler().fit_transform(r)
            tmp = r_std - matrix_std
            tmp = np.matrix(tmp)
        elif test_norm == 4:
            matrix_normalization = matrix / np.sum(matrix) * self.data.user_use
            r_normalization = r / np.sum(r) * self.data.user_use
            tmp = r_normalization - matrix_normalization
        idx_small = np.argwhere((tmp < -1e-9))
        idx_large = np.argwhere((tmp > 1e-9))
        tmp_small = np.zeros(([l, l]))
        tmp_large = np.zeros([l, l])
        for idx in idx_small:
            index = idx[0], idx[1]
            tmp_small[index] = tmp[index]
        for idx in idx_large:
            index = idx[0], idx[1]
            tmp_large[index] = tmp[index]
        tmp_small = tmp_small[tmp_small < 0]
        tmp_large = tmp_large[tmp_large > 0]
        t_small = tmp_small.flatten()
        t_large = tmp_large.flatten()


        min_kth = t_small[np.argpartition(t_small, int(0.95 * len(t_small)))[int(0.95 * len(t_small))]] if len(t_small) else 0
        max_kth = t_large[np.argpartition(t_large, -int(0.95 * len(t_large)))[-int(0.95 * len(t_large))]] if len(t_large) else 10000

        small = np.sort(t_small)
        large = -np.sort(-t_large)


        idx_small = np.argwhere((tmp < min_kth))
        idx_large = np.argwhere((tmp > max_kth))

        matrix_delta = matrix.copy()
        interval = (upper - lower)

        test_unit = 1
        level_Unit = [1 - math.log(i) / math.log(20) for i in range(20, 0, -1)]

        level_Unit_small = [math.log(i) / math.log(len(small)) for i in range(len(small), 0, -1)]
        level_Unit_large = [math.log(i) / math.log(len(large)) for i in range(len(large), 0, -1)]

        for idx in idx_small:
            index = idx[0], idx[1]
            level[index] -= 1 if level[index] > 1 else 0
            if test_unit == 1:
                matrix_delta[index] = matrix[index] - 0.005 * level_Unit_small[np.argwhere(small == tmp[index])[0, 0]]

            if test_unit == 2:
                matrix_delta[index] = interval[index] * level_Unit[int(level[index])]


        for idx in idx_large:
            index = idx[0], idx[1]
            level[index] += 1 if level[index] < 20 else 0
            if test_unit == 1:
                matrix_delta[index] = matrix[index] + 0.005 * level_Unit_large[np.argwhere(large == tmp[index])[0, 0]]

            if test_unit == 2:
                matrix_delta[index] = interval[index] * level_Unit[int(level[index])]

        if (singletons_row == singletons_col):
            matrix_delta[np.diag_indices_from(matrix_delta)] = 0
        matrix_delta = matrix_delta.astype(np.float32)

        return matrix_delta, level

    def build_user_matrix(self, data, singletons_row, singletons_col, length):


        data_temp = []
        for data_j in data:
            user_mat = np.zeros([length, length])
            if (singletons_row == singletons_col):
                sinlgetons = singletons_row
                if data_j and type(data_j[0]) == tuple:
                    for itemset in data_j:
                        if len(set(itemset).intersection(sinlgetons)) == 2:
                            user_mat[sinlgetons.index(itemset[0])][sinlgetons.index(itemset[1])] = \
                                user_mat[sinlgetons.index(itemset[1])][sinlgetons.index(itemset[0])] = 1
                else:
                    data_j = set(data_j)
                    data_j = data_j.intersection(sinlgetons)
                    cc = list(itertools.combinations(data_j, 2))
                    for item in cc:
                        user_mat[sinlgetons.index(item[0])][sinlgetons.index(item[1])] = \
                            user_mat[sinlgetons.index(item[1])][sinlgetons.index(item[0])] = 1
            else:
                if data_j and type(data_j[0]) == tuple:
                    for itemset in data_j:
                        if (len(itemset) == 2 and itemset[0] in singletons_row and itemset[1] in singletons_col):
                            user_mat[singletons_row.index(itemset[0])][singletons_col.index(itemset[1])] = 1
                else:
                    userdata = set(data_j)
                    userdata_row = userdata.intersection(singletons_row)
                    userdata_col = userdata.intersection(singletons_col)
                    for r in userdata_row:
                        for c in userdata_col:
                            user_mat[singletons_row.index(r)][singletons_col.index(c)] = 1

            data_temp.append(user_mat)

        return data_temp

    def domain(self, U, V):
        length = U[0].size
        U = np.mat(U)
        s = np.zeros((length, length), dtype=object)
        for i in range(length):
            for j in range(length):
                corr = np.dot(U[:, i], [V[j]])
                corr[np.diag_indices_from(corr)] = 0
                max_ = min_ = 0
                if len(V[0]) == 1:
                    corr_ = np.reshape(corr, (1, U[:, 0].size ** 2)).tolist()
                else:
                    corr_ = np.reshape(corr, (1, len(V[0]) ** 2)).tolist()
                for x in corr_[0]:
                    x = np.float32(x)
                    if x > 0:
                        max_ += x
                    if x < 0:
                        min_ += x
                s[i, j] = (max_, min_)

        return s

    def domain_limit(self, U, V, length_limit):
        length = U[0].size
        limit = min(length ** 2, length_limit ** 2)
        U = np.mat(U)
        s = np.zeros((length, length), dtype=object)
        for i in range(length):
            for j in range(length):
                corr = np.dot(U[:, i], [V[j]])
                corr[np.diag_indices_from(corr)] = 0
                max_ = min_ = 0
                if len(V[0]) == 1:
                    corr_ = np.reshape(corr, (1, U[:, 0].size ** 2)).tolist()
                else:
                    corr_ = np.reshape(corr, (1, len(V[0]) ** 2)).tolist()
                UV = corr_[0].copy()
                UV.sort(reverse=True)
                for x in UV[:limit]:
                    x = np.float32(x)
                    if x > 0:
                        max_ += x
                for x in UV[-limit:]:
                    x = np.float32(x)
                    if x < 0:
                        min_ += x
                s[i, j] = (max_, min_)
        # s is a matrix of tuples which include the max and min value of S_ij
        return s

    def domain2(self, U, V, length_limit, clip_coefficient=1.0):
        length = U[0].size
        limit = min(length ** 2, length_limit ** 2)
        U = np.mat(U)
        s = np.zeros((length), dtype=object)
        for i in range(length):
            corr = np.dot(U[:, i], [V[i]])
            corr[np.diag_indices_from(corr)] = 0

            max_ = min_ = 0
            if len(V[0]) == 1:
                corr_ = np.reshape(corr, (1, U[:, 0].size ** 2)).tolist()
            else:
                corr_ = np.reshape(corr, (1, len(V[0]) ** 2)).tolist()
            UV = corr_[0].copy()
            UV.sort(reverse=True)
            # print(UV)
            for x in UV[:limit]:
                x = np.float32(x)
                if x > 0:
                    max_ += x
            for x in UV[-limit:]:
                x = np.float32(x)
                if x < 0:
                    min_ += x
            s[i] = (max_ * clip_coefficient, min_ * clip_coefficient)
        # s is a matrix of tuples which include the max and min value of S_ij
        return s

    # from the estimated S_ij^ recovering the estimated items matrix M^
    def recover_frequent_itemset(self, points, U, V):
        S = points.copy()
        # S = np.diagonal(S)
        S = np.diag(S)
        return U @ S @ V

    # calculate the S_ii for each user (i 0~r)
    def report_diag(self, U, V, data):
        report = []
        for item in data:
            S = (U.T @ item) @ V.T
            S = np.diagonal(S)
            report.append(S.tolist())
        return report

    def svdfo_build_results(self, results_itemset):


        results_items = {}
        for key in results_itemset.keys():
            item = (key[0] - 1) * self.data.dict_size + key[1]
            results_items[tuple([item])] = results_itemset[key]
        return results_items


