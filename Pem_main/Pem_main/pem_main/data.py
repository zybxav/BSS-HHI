import abc
from collections import Counter
import pickle
import gzip
from os import path

import numpy as np
import random

class Data_single(object):
    data = None
    heavy_hitters = None
    heavy_hitter_freq = None
    input_file=None
    user_sets=None
    heavy_size=None

    def __init__(self, n_bits,heavy_size,input_file):

        random.seed(42)
        self.n_bits = n_bits
        self.heavy_size = heavy_size
        self.input_file=input_file
        self.heavy_size=heavy_size
        self.user_sets=self.load_data(input_file)
        print(len(self.user_sets))
        self.data = [random.choice(user_set) for user_set in self.user_sets if len(user_set) > 0]
        self.compute_heavy_hitters()

    def compute_heavy_hitters(self):
        counter = Counter()
        for user_set in self.user_sets:
            counter.update(user_set)

        most_common = counter.most_common(self.heavy_size) 
        self.heavy_hitters = [item for item, _ in most_common] 
        self.heavy_hitter_freq = [freq for _, freq in most_common] 

    def load_data(self, path):
        user_sets = []
        with open(path, 'r') as f:
            for line in f:
                items = list(map(int, line.strip().split()))
                if len(items) > 0:

                    chosen_item = random.choice(items)
                    user_sets.append([chosen_item])

        sample_size = 100000
        if len(user_sets) > sample_size:
            random.seed(42)
            user_sets = random.sample(user_sets, sample_size)

        return user_sets
    
    
    def suffix_tally(self, low, high, cand_dict, shift_num):
        mask = (1 << shift_num) - 1
        buckets = np.zeros(max(cand_dict.values()) + 1, dtype=int)
        
        for i in range(int(low), int(high)):
            cur_value = self.data[i] & mask
            if cur_value in cand_dict:
                buckets[cand_dict[cur_value]] += 1
        return buckets
