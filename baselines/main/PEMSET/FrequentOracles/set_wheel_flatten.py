# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import xxhash
import heapq
import struct
import collections
import os

def flat_data(X, N, c, d, flat_num):
    X1 = []
    for i in range(N):
        X1.append([])
        temp_data = collections.Counter(X[i])
        if max(temp_data.values()) > 1:
            data_1 = filter(lambda x: x[1] == 1, temp_data.items())
            for t1_1, t1_2 in data_1:
                X1[i].append(t1_1)
            for j in range(2, max(temp_data.values()) + 1):
                data_j = filter(lambda x: x[1] == j, temp_data.items())
                for tj_1, tj_2 in data_j:
                    if j < flat_num:
                        for t_j in range(j):
                            t_index = X[i].index(tj_1)
                            X1[i].append(X[i][t_index] + (j - t_j - 1) * d)
                    else:
                        for t_j in range(j):
                            t_index = X[i].index(tj_1)
                            if j - t_j >= flat_num:
                                X1[i].append(X[i][t_index] + (flat_num - 1) * d)
                            else:
                                X1[i].append(X[i][t_index] + (j - t_j - 1) * d)
        else:
            for j in range(c):
                X1[i].append(X[i][j])
    return X1

def wheel_perturb_set_data_flatten(X, N, c, epsilon):
    max_int_32 = (1 << 32) - 1
    Y = [0 for col in range(N)]
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    omega = c * temp_p * s + (1 - c * temp_p)
    for i in range(N):
        V = [0 for col in range(c)]
        for j in range(c):
            V[j] = xxhash.xxh32_intdigest(str(X[i][j]), seed=i) / max_int_32
        bSize = math.ceil(1 / temp_p)
        lef = [0 for col in range(bSize)]
        rig = [0 for col in range(bSize)]
        for b in range(bSize):
            lef[b] = min((b + 1) * temp_p, 1.0)
            rig[b] = b * temp_p
        for v in V:
            temp_b = math.ceil(v / temp_p) - 1
            lef[temp_b] = min(v, lef[temp_b])
            if temp_b < math.ceil(1 / temp_p) - 1:
                rig[temp_b + 1] = max(v + temp_p, rig[temp_b + 1])
            else:
                rig[0] = max(v + temp_p - 1, rig[0])
        temp_rig0 = rig[0]
        for b in range(bSize - 1):
            lef[b] = max(lef[b], rig[b])
            rig[b] = rig[b + 1]
        lef[bSize - 1] = max(lef[bSize - 1], rig[bSize - 1])
        rig[bSize - 1] = temp_rig0 + 1.0
        ll = 0.0
        for b in range(bSize):
            ll = ll + rig[b] - lef[b]
        r = np.random.random_sample()
        a = 0.0
        for b in range(bSize):
            a = a + s * (rig[b] - lef[b]) / omega
            if a > r:
                z = rig[b] - (a - r) * omega / s
                break
            a = a + (omega - ll * s) * (lef[(b + 1) % round(bSize)] + math.floor((b + 1) * temp_p) - rig[b]) / ((1 - ll) * omega)
            if a > r:
                z = lef[(b + 1) % bSize] - (a - r) * (1 - ll) * omega / (omega - ll * s)
                break
        z = z % 1.0
        Y[i] = z
    return Y

def wheel_aggregate_set_data_flatten(Y, N, c, epsilon, D, d, flat_num):
    max_int_32 = (1 << 32) - 1
    k = len(D)
    Estimate_Dist = [0 for col in range(k)]
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    for i in range(N):
        z = Y[i]
        for j in range(k):
            for t_flat in range(flat_num):
                x = D[j] + t_flat * d
                v = xxhash.xxh32_intdigest(str(x), seed=i) / max_int_32
                if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                    Estimate_Dist[j] += 1
    pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
    pf = temp_p
    for i in range(k):
        Estimate_Dist[i] = 1 / N * (Estimate_Dist[i] - N * pf * flat_num) / (pt - pf)
    return Estimate_Dist

def wheel_aggregate_set_data_file_flatten(Y, N, c, epsilon, r, d, flat_num):
    max_int_32 = (1 << 32) - 1
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
    pf = temp_p
    file_d_name = '../../temp/wheel_flatten/D' + str(r) + '.txt'
    file_dist_name = '../../temp/wheel_flatten/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'w')
    str_data = file_d.readline().replace('\n', '')
    while str_data:
        t_count = 0
        for t_flat in range(flat_num):
            x_data = int(str_data) + t_flat * d
            for i in range(N):
                z = Y[i]
                v = xxhash.xxh32_intdigest(str(x_data), seed=i) / max_int_32
                if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                    t_count += 1
        t_count = 1 / N * (t_count - N * pf * flat_num) / (pt - pf)
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(t_count))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0

def set_wheel_flatten(X, N, c, epsilon, D, d, flat_num):
    X1 = flat_data(X, N, c, d, flat_num)
    Y = wheel_perturb_set_data_flatten(X1, N, c, epsilon)
    EstimateDist_wheel = wheel_aggregate_set_data_flatten(Y, N, c, epsilon, D, d, flat_num)
    return EstimateDist_wheel

def set_wheel_file_flatten(X, N, c, epsilon, r, d, flat_num):
    X1 = flat_data(X, N, c, d, flat_num)
    Y = wheel_perturb_set_data_flatten(X1, N, c, epsilon)
    wheel_aggregate_set_data_file_flatten(Y, N, c, epsilon, r, d, flat_num)
    return 0
