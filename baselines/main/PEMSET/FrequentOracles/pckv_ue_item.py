# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import xxhash
import heapq
import struct
import collections
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

def set_to_kv(X, N, c):
    global max_v, min_v
    k_v = []
    for i in range(N):
        k_v.append([])
    for i in range(N):
        temp_data = collections.Counter(X[i])
        t_keys = sorted(temp_data.keys())
        temp = dict(temp_data)
        for j in range(len(t_keys)):
            t_key = t_keys[j]
            t_value = temp[t_key]
            k_v[i].append([t_key, t_value])
    for i in range(N):
        for j in range(len(k_v[i])):
            k_v[i][j][1] = 2 * k_v[i][j][1] / (c - 1) + (-c - 1) / (c - 1)
    return k_v

def P_S(X, l, N, d):
    k_v = [[0 for col in range(2)] for row in range(N)]
    for i in range(N):
        S = len(X[i])
        b = S / max(S, l)
        rnd = np.random.random()
        if rnd < b:
            tmp = random.sample(X[i], 1)
            k_ps = tmp[0][0]
            v_ps = tmp[0][1]
        else:
            v_ps = 0
            k_ps = random.randint(d, d + l - 1)
        p = (1 + v_ps) / 2
        rnd = np.random.random()
        if rnd < p:
            v_ps = 1
        else:
            v_ps = -1
        k_v[i][0] = k_ps
        k_v[i][1] = v_ps
    return k_v

def perturb_item(x, k, v, epsilon):
    epsilon1 = math.log((math.exp(epsilon) + 1) / 2)
    epsilon2 = epsilon
    a = 1 / 2
    b = 1 / (math.exp(epsilon1) + 1)
    p = math.exp(epsilon2) / (math.exp(epsilon2) + 1)
    r = np.random.random()
    if x == k:
        if r < a * p:
            t_value = v
        elif r < a:
            t_value = -v
        else:
            t_value = 0
    elif r < b / 2:
        t_value = 1
    elif r < b:
        t_value = -1
    else:
        t_value = 0
    return t_value

def aggregate_item(n1, n2, l, epsilon, N, c):
    epsilon1 = math.log((math.exp(epsilon) + 1) / 2)
    epsilon2 = epsilon
    a = 1 / 2
    b = 1 / (math.exp(epsilon1) + 1)
    p = math.exp(epsilon2) / (math.exp(epsilon2) + 1)
    t1 = a * p - b / 2
    t2 = a * (1 - p) - b / 2
    A = np.array([[t1, t2], [t2, t1]])
    A1 = np.linalg.inv(A)
    fk = l * ((n1 + n2) / N - b) / (a - b)
    if fk < 1 / N:
        fk = 1 / N
    elif fk > 1:
        fk = 1
    t = np.dot(A1, [[n1 - N * b / 2], [n2 - N * b / 2]])
    t_n1 = float(t[0])
    if t_n1 < 0:
        t_n1 = 0
    elif t_n1 > N * fk / l:
        t_n1 = N * fk / l
    t_n2 = float(t[1])
    if t_n2 < 0:
        t_n2 = 0
    elif t_n2 > N * fk / l:
        t_n2 = N * fk / l
    mk = l * (t_n1 - t_n2) / (N * fk)
    tmk = (c - 1) * mk / 2 + (c + 1) / 2
    return (fk, tmk)

def PCKV_UE(X, N, c, epsilon, D, d):
    global max_v, min_v
    len_D = len(D)
    X_t = set_to_kv(X, N, c)
    k_v = P_S(X_t, c, N, d)
    EstimateDist = [0 for col in range(len_D)]
    for i in range(len_D):
        n1 = 0
        n2 = 0
        t_key = D[i]
        for j in range(N):
            k = k_v[j][0]
            v = k_v[j][1]
            t_value = perturb_item(t_key, k, v, epsilon)
            if t_value == 1:
                n1 += 1
            elif t_value == -1:
                n2 += 1
        fk, tmk = aggregate_item(n1, n2, c, epsilon, N, c)
        EstimateDist[i] = fk * tmk
    return EstimateDist

def PCKV_UE_file(X, N, c, epsilon, r, d):
    file_d_name = '../../temp/pckv_ue/D' + str(r) + '.txt'
    file_dist_name = '../../temp/pckv_ue/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'w')
    str_data = file_d.readline().replace('\n', '')
    X_t = set_to_kv(X, N, c)
    k_v = P_S(X_t, c, N, d)
    while str_data:
        t_key = int(str_data)
        n1 = 0
        n2 = 0
        for j in range(N):
            k = k_v[j][0]
            v = k_v[j][1]
            t_value = perturb_item(t_key, k, v, epsilon)
            if t_value == 1:
                n1 += 1
            elif t_value == -1:
                n2 += 1
        fk, tmk = aggregate_item(n1, n2, c, epsilon, N, c)
        EstimateDist = fk * tmk
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(EstimateDist))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0
