# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import xxhash
import heapq
import struct
import collections
from .OLH import OLH, OLH_file
from .OLH_sample import OLH_sample, OLH_sample_file
from .set_wheel import set_wheel, set_wheel_file
from .pckv_ue_item import PCKV_UE, PCKV_UE_file
from .read_data_dist import read_data

def data_to_prefix(X, N, g, c, m, k):
    Ni = [0 for col in range(g + 1)]
    si = [0 for col in range(g + 1)]
    prefix_X = [[0 for col in range(c)] for row in range(N)]
    r_N = math.ceil(N / g)
    for i in range(1, g + 1):
        r_start = (i - 1) * r_N
        b_start = i - 1
        if i == g:
            r_stop = N
        else:
            r_stop = i * r_N
        s = math.ceil(math.log2(k)) + math.ceil(i * (m - math.ceil(math.log2(k))) / g)
        si[i] = s
        for j in range(r_start, r_stop):
            for t_j in range(c):
                prefix_X[j][t_j] = int('{:064b}'.format(int(X[j][t_j]))[0:s], 2)
    return (prefix_X, si)

def split_prefix_X(prefix_X, r, g, N):
    r_N = math.ceil(N / g)
    r_start = (r - 1) * r_N
    if r == g:
        r_stop = N
    else:
        r_stop = r * r_N
    t_X = prefix_X[r_start:r_stop]
    return (t_X, r_stop - r_start)

def construct_D(Ct, k, m, r, g):
    sr = math.ceil(math.log2(k)) + math.ceil(r * (m - math.ceil(math.log2(k))) / g)
    if r == 1:
        sr1 = math.ceil(math.log2(k))
    else:
        sr1 = math.ceil(math.log2(k)) + math.ceil((r - 1) * (m - math.ceil(math.log2(k))) / g)
    s_len = sr - sr1
    sl = int(math.pow(2, s_len))
    l = k * sl
    s = [0 for col in range(sl)]
    for i in range(sl):
        s[i] = i
    D = [0 for col in range(l)]
    p = 0
    for i in range(k):
        for j in range(sl):
            t_num = '{:b}'.format(int(Ct[i])) + '{:b}'.format(int(s[j])).zfill(s_len)
            D[p] = int(t_num, 2)
            p += 1
    return D

def construct_D_file(Ct, k, m, r, g):
    sr = math.ceil(math.log2(k)) + math.ceil(r * (m - math.ceil(math.log2(k))) / g)
    if r == 1:
        sr1 = math.ceil(math.log2(k))
    else:
        sr1 = math.ceil(math.log2(k)) + math.ceil((r - 1) * (m - math.ceil(math.log2(k))) / g)
    s_len = sr - sr1
    sl = int(math.pow(2, s_len))
    s = [0 for col in range(sl)]
    for i in range(sl):
        s[i] = i
    filename = 'temp/D' + str(r) + '.txt'
    f1 = open(filename, 'w')
    for i in range(k):
        for j in range(sl):
            t_num = '{:b}'.format(int(Ct[i])) + '{:b}'.format(int(s[j])).zfill(s_len)
            f1.write(str(int(t_num, 2)))
            f1.write('\n')
    return 0

def construct_C(C1, k, D):
    top_k_data = [0 for col in range(k)]
    top_k_dist = [0 for col in range(k)]
    for i in range(k):
        top_k_dist[i] = max(C1)
        top_k_data[i] = D[C1.index(max(C1))]
        C1[C1.index(max(C1))] = -100
    return (top_k_data, top_k_dist)

def construct_C_file(k, r):
    top_k_data = [0 for col in range(k)]
    top_k_dist = [0 for col in range(k)]
    file_d_name = 'temp/D' + str(r) + '.txt'
    file_dist_name = 'temp/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'r')
    str_data = file_d.readline().replace('\n', '')
    str_dist = file_dist.readline().replace('\n', '')
    while str_dist:
        dist = float(str_dist)
        data = int(str_data)
        t_min = min(top_k_dist)
        if dist > t_min:
            p_min = top_k_dist.index(min(top_k_dist))
            top_k_dist[p_min] = dist
            top_k_data[p_min] = data
        str_dist = file_dist.readline()
        str_data = file_d.readline()
    file_d.close()
    file_dist.close()
    return (top_k_data, top_k_dist)

def cal_f1(ct, cg):
    c = []
    for i in range(len(ct)):
        if ct[i] in cg:
            c.append(ct[i])
    lc = len(c)
    p = lc / len(ct)
    r = lc / len(cg)
    if p == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    print(c)
    return f1

def cal_ncr(ct, cg, dist_ct, dist_cg, k):
    dist_t1 = [0 for col in range(k)]
    dist_t2 = [0 for col in range(k)]
    for i in range(k):
        dist_t1[i] = dist_ct[i]
        dist_t2[i] = dist_cg[i]
    order_ct = []
    order_cg = []
    ncr = 0
    for i in range(k):
        t_index1 = dist_t1.index(max(dist_t1))
        order_ct.append(ct[t_index1])
        dist_t1[t_index1] = 0
        t_index2 = dist_t2.index(max(dist_t2))
        order_cg.append(cg[t_index2])
        dist_t2[t_index2] = 0
    for i in range(k):
        t = order_ct[i]
        if t in order_cg:
            ncr = ncr + (k - order_cg.index(t))
    ncr = ncr / (k * (k + 1) / 2)
    return ncr

def PEM_OLH(prefix_X, k, m, g, c, epsilon, N):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    if l_s < 20:
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
            else:
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
    else:
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                OLH_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_dist = construct_C_file(k, i)
            else:
                construct_D_file(Ct, k, m, i, g)
                OLH_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_dist = construct_C_file(k, i)
    return (Ct, OLH_dist)

def PEM_OLH_sample(prefix_X, k, m, g, c, epsilon, N):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    if l_s < 20:
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH_sample = OLH_sample(t_X, Ni, c, epsilon, D)
                Ct, OLH_sample_dist = construct_C(EstimateDist_OlH_sample, k, D)
            else:
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH_sample = OLH_sample(t_X, Ni, c, epsilon, D)
                Ct, OLH_sample_dist = construct_C(EstimateDist_OlH_sample, k, D)
    else:
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                OLH_sample_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_sample_dist = construct_C_file(k, i)
            else:
                construct_D_file(Ct, k, m, i, g)
                OLH_sample_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_sample_dist = construct_C_file(k, i)
    return (Ct, OLH_sample_dist)

def PEM_wheel(prefix_X, k, m, g, c, epsilon, N):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    if l_s < 20:
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_wheel = set_wheel(t_X, Ni, c, epsilon, D)
                Ct, wheel_dist = construct_C(EstimateDist_wheel, k, D)
            else:
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_wheel = set_wheel(t_X, Ni, c, epsilon, D)
                Ct, wheel_dist = construct_C(EstimateDist_wheel, k, D)
    else:
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                set_wheel_file(t_X, Ni, c, epsilon, i)
                Ct, wheel_dist = construct_C_file(k, i)
            else:
                construct_D_file(Ct, k, m, i, g)
                set_wheel_file(t_X, Ni, c, epsilon, i)
                Ct, wheel_dist = construct_C_file(k, i)
    return (Ct, wheel_dist)

def PEM_PCKV_UE(prefix_X, k, m, g, c, epsilon, N, si):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    if l_s < 20:
        for i in range(1, g + 1):
            s = pow(2, si[i])
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_PCKV_UE = PCKV_UE(t_X, Ni, c, epsilon, D, s)
                Ct, PCKV_UE_dist = construct_C(EstimateDist_PCKV_UE, k, D)
            else:
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_PCKV_UE = PCKV_UE(t_X, Ni, c, epsilon, D, s)
                Ct, PCKV_UE_dist = construct_C(EstimateDist_PCKV_UE, k, D)
    else:
        for i in range(1, g + 1):
            s = pow(2, si[i]) - 1
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                PCKV_UE_file(t_X, Ni, c, epsilon, i, s)
                Ct, PCKV_UE_dist = construct_C_file(k, i)
            else:
                construct_D_file(Ct, k, m, i, g)
                PCKV_UE_file(t_X, Ni, c, epsilon, i, s)
                Ct, PCKV_UE_dist = construct_C_file(k, i)
    return (Ct, PCKV_UE_dist)
