# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import xxhash
import heapq
import struct
import collections

def set_to_single(X, N, c):
    S_X = [0 for col in range(N)]
    for i in range(N):
        S_X[i] = X[i][np.random.randint(c)]
    return S_X

def OLH_Perturb_sample(S_X, N, epsilon):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    Y = [0 for col in range(N)]
    for i in range(N):
        Y[i] = xxhash.xxh32(str(S_X[i]), seed=i).intdigest() % g
        t = np.random.random()
        if t > p:
            temp = np.random.randint(g - 1)
            if temp == Y[i]:
                temp = g - 1
            Y[i] = temp
    return Y

def OLH_Aggregate_sample(Y, N, c, epsilon, D):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    len_D = len(D)
    Z = [0 for col in range(len_D)]
    for i in range(len_D):
        t_count = 0
        for j in range(N):
            temp = xxhash.xxh32(str(D[i]), seed=j).intdigest() % g
            if temp == Y[j]:
                t_count += 1
        Z[i] = c * (t_count / N - q) / (p - q)
    return Z

def OLH_Aggregate_sample_file(Y, N, c, epsilon, r):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    file_d_name = '../../temp/olh_sample/D' + str(r) + '.txt'
    file_dist_name = '../../temp/olh_sample/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'w')
    str_data = file_d.readline().replace('\n', '')
    while str_data:
        t_count = 0
        for i in range(N):
            temp = xxhash.xxh32(str_data, seed=i).intdigest() % g
            if temp == Y[i]:
                t_count += 1
        t_count = c * (t_count / N - q) / (p - q)
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(t_count))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0

def OLH_sample(X, N, c, epsilon, D):
    S_X = set_to_single(X, N, c)
    Y = OLH_Perturb_sample(S_X, N, epsilon)
    EstimateDist_OlH = OLH_Aggregate_sample(Y, N, c, epsilon, D)
    return EstimateDist_OlH

def OLH_sample_file(X, N, c, epsilon, r):
    S_X = set_to_single(X, N, c)
    Y = OLH_Perturb_sample(S_X, N, epsilon)
    OLH_Aggregate_sample_file(Y, N, c, epsilon, r)
    return 0
