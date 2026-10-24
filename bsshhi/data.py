# -*- coding: utf-8 -*-
import numpy as np

class Data(object):

    def __init__(self, rows, name, domain):
        self.data = list(rows)
        self.data_name = name
        self.dict_size = int(domain)
        self.user_use = len(self.data)

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

    def test_length_cand(self, low, high, cand_list, limit, start_limit=0):
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
            if value <= start_limit:
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

    def true_item_cand(self, top_k):
        results = {}
        for user_items in self.data:
            if len(user_items) > 0 and isinstance(user_items[0], tuple):
                items = set()
                for tup in user_items:
                    items.update(tup)
            else:
                items = [item for item in user_items if item != -1]
            for item in items:
                key = (item,)
                results[key] = results.get(key, 0) + 1
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_k_results = dict(sorted_results[:top_k])
        return top_k_results
