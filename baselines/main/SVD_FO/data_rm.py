import ast
import heapq
import itertools
import math
import numpy as np
from bsshhi.data import Data as CoreData
class Data(CoreData):

    def test_single(self, low, high):
        results = np.zeros(self.dict_size + 1, dtype=np.int_)
        for i in range(low, high):
            if len(self.data[i]) == 0:
                continue
            if type(self.data[i][0]) is tuple:
                temp = []
                for tup in self.data[i]:
                    temp.extend(list(tup))
                temp = list(set(temp))
                temp.sort()
                rand_index = np.random.randint(len(temp))
                value = temp[rand_index]
                if value == -1:
                    continue
                results[value] += 1
            else:
                rand_index = np.random.randint(len(self.data[i]))
                value = self.data[i][rand_index]
                if value == -1:
                    continue
                results[value] += 1
        return results

    def test_length_cand(self, low, high, cand_list, limit, start_limit=0, include_zero=False):
        results = np.zeros(limit - start_limit + 1, dtype=np.int_)
        cand_set = set(cand_list)
        for i in range(low, high):
            X = self.data[i]
            if type(X[0]) is tuple:
                value = 0
                for itemset in X:
                    for item in itemset:
                        if item in cand_set:
                            value += 1
            else:
                value = 0
                for item in X:
                    if item in cand_set:
                        value += 1
            if value < start_limit or (value == start_limit and (not include_zero)):
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    def test_singleton_cand_limit(self, low, high, key_dict, singleton_set, length_limit):
        results = np.zeros(len(singleton_set) + 1, dtype=np.int_)
        for i in range(low, high):
            values = []
            x = self.data[i]
            if type(x[0]) is tuple:
                for itemset in x:
                    for item in itemset:
                        if item in singleton_set:
                            values.append(item)
            else:
                for item in x:
                    if item in singleton_set:
                        values.append(item)
            if len(values) > length_limit:
                rand_index = np.random.randint(len(values))
                result = key_dict[values[rand_index],]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(singleton_set)
                if rand_index < len(values):
                    result = key_dict[values[rand_index],]
            results[result] += 1
        return results

    def test_length_itemset(self, low, high, cand_dict, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int_)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))
        for i in range(low, high):
            if type(self.data[i][0]) is tuple:
                value = 0
                for itemset in self.data[i]:
                    if itemset in cand_dict:
                        value += 1
                if value <= start_limit:
                    continue
                if value > limit:
                    value = start_limit
                results[value - start_limit] += 1
            else:
                current_set = singleton_set.intersection(set(self.data[i]))
                if len(current_set) == 0:
                    continue
                value = 0
                for cand in cand_dict:
                    if set(cand) <= current_set:
                        value += 1
                if value <= start_limit:
                    continue
                if value > limit:
                    value = start_limit
                results[value - start_limit] += 1
        return results

    def candidate_list2mat(self, low, high, cand_dict, length):
        data = self.data[low:high]
        mat = np.zeros([length, length])
        for i in range(high - low):
            if len(self.data[i]) and type(self.data[i][0]) == tuple:
                for itemset in data[i]:
                    if len(set(itemset).intersection(cand_dict)) == 2:
                        mat[cand_dict.index(itemset[0])][cand_dict.index(itemset[1])] = mat[cand_dict.index(itemset[1])][cand_dict.index(itemset[0])] = mat[cand_dict.index(itemset[0])][cand_dict.index(itemset[1])] + 1
            else:
                candidate = set(data[i])
                candidate = candidate.intersection(cand_dict)
                cc = list(itertools.combinations(candidate, 2))
                for item in cc:
                    mat[cand_dict.index(item[0])][cand_dict.index(item[1])] = mat[cand_dict.index(item[1])][cand_dict.index(item[0])] = mat[cand_dict.index(item[0])][cand_dict.index(item[1])] + 1
        return (data, mat)

    def candidate_list2mat_block(self, low, high, cand_row_dict, cand_col_dict, row_length, col_length):
        data = self.data[low:high]
        mat = np.zeros([row_length, col_length])
        for i in range(high - low):
            if len(self.data[i]) and type(self.data[i][0]) == tuple:
                for itemset in data[i]:
                    if itemset[0] in cand_row_dict and itemset[1] in cand_col_dict:
                        mat[cand_row_dict.index(itemset[0])][cand_col_dict.index(itemset[1])] += 1
                        mat[cand_col_dict.index(itemset[1])][cand_row_dict.index(itemset[0])] += 1
            else:
                candidate = set(data[i])
                candidate_row = candidate.intersection(cand_row_dict)
                candidate_col = candidate.intersection(cand_col_dict)
                cc = list(itertools.product(candidate_row, candidate_col))
                for item in cc:
                    mat[cand_row_dict.index(item[0])][cand_col_dict.index(item[1])] += 1
            if cand_row_dict == cand_col_dict:
                unique_items_in_user = set(itertools.chain(*data[i])) if len(self.data[i]) and type(self.data[i][0]) == tuple else set(data[i])
                for item in unique_items_in_user:
                    if item in cand_row_dict:
                        mat[cand_row_dict.index(item)][cand_row_dict.index(item)] += 1
        return (data, mat)

    def test_itemsets_cand_limit(self, low, high, cand_dict, length_limit):
        buckets = np.zeros(len(cand_dict) + 1, dtype=np.int_)
        if type(self.data[0][0]) is tuple:
            for i in range(low, high):
                subset_count = 0
                subset_indices = []
                for itemset in self.data[i]:
                    if itemset in cand_dict:
                        subset_count += 1
                        subset_indices.append(cand_dict[itemset])
                if subset_count > length_limit:
                    rand_index = np.random.randint(subset_count)
                    result = subset_indices[rand_index]
                else:
                    rand_index = np.random.randint(length_limit)
                    result = len(cand_dict)
                    if rand_index < subset_count:
                        result = subset_indices[rand_index]
                buckets[result] += 1
        else:
            singleton_set = set()
            for cand in cand_dict:
                singleton_set = singleton_set.union(set(cand))
            for i in range(low, high):
                current_set = singleton_set.intersection(set(self.data[i]))
                if len(current_set) == 0:
                    continue
                subset_count = 0
                subset_indices = []
                for cand in cand_dict:
                    if set(cand) <= current_set:
                        subset_count += 1
                        subset_indices.append(cand_dict[cand])
                if subset_count > length_limit:
                    rand_index = np.random.randint(subset_count)
                    result = subset_indices[rand_index]
                else:
                    rand_index = np.random.randint(length_limit)
                    result = len(cand_dict)
                    if rand_index < subset_count:
                        result = subset_indices[rand_index]
                buckets[result] += 1
        return buckets

    def true_item_cand(self, top_k):
        results = {}
        for user_items in self.data:
            if len(user_items) > 0 and isinstance(user_items[0], tuple):
                items = set()
                for tup in user_items:
                    items.update(tup)
            else:
                dummy_item_id = getattr(self, 'dummy_item_id', None)
                items = [item for item in user_items if item != -1 and item != dummy_item_id]
            for item in items:
                key = (item,)
                results[key] = results.get(key, 0) + 1
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_k_results = dict(sorted_results[:top_k])
        return top_k_results

    def true_itemset_cand(self, low, high, top_k):
        results = {}
        dic = None
        file_path = '../middleware/dic' + self.data_name + str(high) + '.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for j in range(low, high):
                if type(self.data[j]) == int:
                    print(j, self.data[j])
                elif len(self.data[j]) and type(self.data[j][0]) == tuple:
                    for itemsets in self.data[j]:
                        temp = tuple(itemsets)
                        if temp in results:
                            results[temp] += 1
                        else:
                            results[temp] = 1
                else:
                    x = set(self.data[j])
                    for c in itertools.combinations(x, 2):
                        if -1 not in tuple(c):
                            temp = (min(c[0], c[1]), max(c[0], c[1]))
                            if temp in results:
                                results[temp] += 1
                            else:
                                results[temp] = 1
                            temp = (max(c[0], c[1]), min(c[0], c[1]))
                            if temp in results:
                                results[temp] += 1
                            else:
                                results[temp] = 1
            temp = {}
            for key in results:
                if results[key] > 0:
                    temp[key] = results[key]
            results = sorted(temp.items(), key=lambda x: x[1], reverse=True)
            results = dict(results)
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_k:
                break
        return new_results

    def true_conf_cand(self, low, high, top_k, top_ks, top_kc):
        results = {}
        rule_results = {}
        singletons = self.true_item_cand(int(top_k))
        itemsets_onlyk = self.true_itemset_cand_onlyk(low, high, top_k, top_ks)
        dic = None
        file_path = '../middleware/dic' + self.data_name + str(high) + '_k_' + str(top_k) + '_ks_' + str(top_ks) + '_conf.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for itemset in itemsets_onlyk.keys():
                rule_results[tuple([itemset[0], itemset[1]])] = itemsets_onlyk[itemset] / singletons[tuple([itemset[0]])]
            results = dict(sorted(rule_results.items(), key=lambda x: x[1], reverse=True))
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_kc:
                break
        return new_results

    def true_itemset_cand_onlyk(self, low, high, top_k, top_ks):
        itemset_results = {}
        itemset_results_onlyk = {}
        singletons = self.true_item_cand(int(top_k))
        itemsets_all = self.true_itemset_cand(low, high, self.dict_size * (self.dict_size - 1))
        for itemset in itemsets_all.keys():
            if tuple([itemset[0]]) in singletons and tuple([itemset[1]]) in singletons:
                itemset_results[itemset] = itemsets_all[itemset]
        itemset_results = dict(sorted(itemset_results.items(), key=lambda x: x[1], reverse=True))
        i = 0
        for key in itemset_results.keys():
            itemset_results_onlyk[key] = itemset_results[key]
            i += 1
            if i == top_ks:
                break
        return itemset_results_onlyk

    def true_conf_cand1(self, low, high, top_k, top_ks, top_kc):
        results = {}
        rule_results = {}
        singletons = self.true_item_cand(int(top_k))
        itemsets_onlyk = self.true_itemset_cand_onlyk(low, high, top_k, top_ks)
        dic = None
        file_path = '../middleware/dic' + self.data_name + str(high) + '_k_' + str(top_k) + '_ks_' + str(top_ks) + '_conf.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for itemset in itemsets_onlyk.keys():
                rule_results[tuple([itemset[0], itemset[1]])] = itemsets_onlyk[itemset] / singletons[tuple([itemset[0]])]
                rule_results[tuple([itemset[1], itemset[0]])] = itemsets_onlyk[itemset] / singletons[tuple([itemset[1]])]
            results = dict(sorted(rule_results.items(), key=lambda x: x[1], reverse=True))
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_kc:
                break
        return new_results

    def true_conf_cand2(self, low, high, top_ks, top_kc):
        results = {}
        rule_results = {}
        singletons = self.true_item_cand(self.dict_size)
        relation_ks = self.true_itemset_cand(low, high, top_ks)
        dic = None
        file_path = '../middleware/conf_' + self.data_name + str(high) + '_ks_' + str(top_ks) + '.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for itemset in relation_ks.keys():
                rule_results[tuple([itemset[0], itemset[1]])] = relation_ks[itemset] / singletons[tuple([itemset[0]])]
            results = dict(sorted(rule_results.items(), key=lambda x: x[1], reverse=True))
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_kc:
                break
        return new_results

    def convert_item2pair(self, low, high):
        res_data = []
        size = self.dict_size
        size = math.ceil(math.sqrt(size))
        for i in range(high - low):
            temp = []
            for item in self.data[i]:
                index1 = item // size + 1
                index2 = item - (index1 - 1) * size
                if index2 == 0:
                    temp.append(tuple([index1 - 1, size]))
                else:
                    temp.append(tuple([index1, index2]))
            res_data.append(temp)
        self.data = res_data
        self.dict_size = size

    def singleton_count(self, low, high):
        results = []
        dist = None
        file_path = '../middleware/sing' + self.data_name + str(high) + '.txt'
        try:
            fileObject = open(file_path, 'r')
            dist = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dist is None:
            results = np.zeros(self.dict_size + 1, dtype=np.int_)
            for i in range(low, high):
                if len(self.data[i]) == 0:
                    continue
                for value in self.data[i]:
                    if type(value) == tuple:
                        for item in value:
                            results[item] += 1
                    else:
                        if value == -1 or value == getattr(self, 'dummy_item_id', None):
                            continue
                        results[value] += 1
            results = results.tolist()
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dist
        return results

    def est_top_itemsets(self, top_k, est_dist):
        ret = []
        results = []
        for i in range(len(est_dist)):
            for j in range(len(est_dist)):
                heapq.heappush(ret, (-est_dist[i][j], tuple([i, j])))
        for i in range(2 * top_k):
            value, t = heapq.heappop(ret)
            results.append(t)
        return results

    def remove_non_top_items(self, data, top_d_items):
        """
        Remove items from the data that are not in the top_d_items list,
        and remap the item IDs to be continuous from 0 to d-1 where d is the size of top_d_items.

        :param data: The original data object, where each element is a user's item list.
        :param top_d_items: The list of top items to retain.

        :return: None (data is modified in place).
        """
        top_d_set = set(top_d_items)
        for i, user_items in enumerate(data.data):
            filtered_items = [item for item in user_items if item in top_d_set]
            data.data[i] = filtered_items
        item_mapping = {item: idx for idx, item in enumerate(top_d_items)}
        for i, user_items in enumerate(data.data):
            remapped_items = [item_mapping[item] for item in user_items]
            data.data[i] = remapped_items
        data.dict_size = len(top_d_items)
