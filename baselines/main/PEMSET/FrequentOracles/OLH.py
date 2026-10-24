# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import xxhash
import heapq
import struct
import collections

def OLH_Perturb(X, N, epsilon, c):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    Y = [[0 for col in range(c)] for row in range(N)]
    for i in range(N):
        for j in range(c):
            Y[i][j] = xxhash.xxh32(str(X[i][j]), seed=i).intdigest() % g
            t = np.random.random()
            if t > p:
                temp = np.random.randint(g - 1)
                if temp == Y[i][j]:
                    temp = g - 1
                Y[i][j] = temp
    return Y

def OLH_Aggregate(Y, N, c, epsilon, D):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    len_D = len(D)
    Z = [0 for col in range(len_D)]
    for i in range(len_D):
        t_count = 0
        for j in range(N):
            temp = xxhash.xxh32(str(D[i]), seed=j).intdigest() % g
            for t_j in range(c):
                if temp == Y[j][t_j]:
                    t_count += 1
        Z[i] = (t_count / N - c * q) / (p - q)
    return Z

def OLH_Aggregate_file(Y, N, c, epsilon, r):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    file_d_name = '../../temp/olh/D' + str(r) + '.txt'
    file_dist_name = '../../temp/olh/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'w')
    str_data = file_d.readline().replace('\n', '')
    while str_data:
        t_count = 0
        for i in range(N):
            temp = xxhash.xxh32(str_data, seed=i).intdigest() % g
            for j in range(c):
                if temp == Y[i][j]:
                    t_count += 1
        t_count = (t_count / N - c * q) / (p - q)
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(t_count))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0

def OLH(X, N, c, epsilon, D):
    Y = OLH_Perturb(X, N, epsilon / c, c)
    EstimateDist_OlH = OLH_Aggregate(Y, N, c, epsilon / c, D)
    return EstimateDist_OlH

def OLH_file(X, N, c, epsilon, r):
    Y = OLH_Perturb(X, N, epsilon / c, c)
    OLH_Aggregate_file(Y, N, c, epsilon / c, r)
    return 0
