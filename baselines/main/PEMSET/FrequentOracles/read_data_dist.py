# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import xxhash
import heapq
import struct
import collections

def read_data(file_name, k):
    top_k_data = [0 for col in range(k)]
    top_k_dist = [0 for col in range(k)]
    temp_data = collections.Counter([])
    int_data = []
    f = open(file_name, 'r')
    str_data = f.readline().replace('\n', '')
    N = 0
    while str_data:
        t = str_data.split(' ')
        int_data.append([])
        for i in range(len(t)):
            int_data[N].append(int(t[i]))
        temp = collections.Counter(int_data[N])
        temp_data = temp_data + temp
        N += 1
        str_data = f.readline().replace('\n', '')
    f.close()
    c = len(t)
    for i in range(k):
        top_k_dist[i] = max(temp_data.values()) / N
        top_k_data[i] = max(temp_data, key=temp_data.get)
        temp_data[max(temp_data, key=temp_data.get)] = 0
    return (int_data, top_k_dist, top_k_data, N, c)
