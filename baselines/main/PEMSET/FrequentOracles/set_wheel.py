# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import xxhash
import heapq
import struct
import collections

def wheel_perturb_set_data(X, N, c, epsilon):
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

def wheel_aggregate_set_data(Y, N, c, epsilon, D):
    max_int_32 = (1 << 32) - 1
    k = len(D)
    Estimate_Dist = [0 for col in range(k)]
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    for i in range(N):
        z = Y[i]
        for j in range(k):
            x = D[j]
            v = xxhash.xxh32_intdigest(str(x), seed=i) / max_int_32
            if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                Estimate_Dist[j] += 1
    pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
    pf = temp_p
    for i in range(k):
        Estimate_Dist[i] = 1 / N * (Estimate_Dist[i] - N * pf) / (pt - pf)
    return Estimate_Dist

def wheel_aggregate_set_data_file(Y, N, c, epsilon, r):
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
        for i in range(N):
            z = Y[i]
            v = xxhash.xxh32_intdigest(str_data, seed=i) / max_int_32
            if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                t_count += 1
        t_count = 1 / N * (t_count - N * pf) / (pt - pf)
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(t_count))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0

def set_wheel(X, N, c, epsilon, D):
    Y = wheel_perturb_set_data(X, N, c, epsilon)
    EstimateDist_wheel = wheel_aggregate_set_data(Y, N, c, epsilon, D)
    return EstimateDist_wheel

def set_wheel_file(X, N, c, epsilon, r):
    Y = wheel_perturb_set_data(X, N, c, epsilon)
    wheel_aggregate_set_data_file(Y, N, c, epsilon, r)
    return 0
