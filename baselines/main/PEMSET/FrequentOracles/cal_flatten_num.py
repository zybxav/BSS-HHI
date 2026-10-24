# -*- coding: utf-8 -*-
import math
from scipy.special import comb
from functools import reduce
import numpy as np

def cal_factorial(a, b):
    if b >= a:
        return reduce(lambda x, y: x * y, range(a, b + 1))
    else:
        return 1

def cal_below_t2(n, m, t):
    sum = 0
    for i in range(0, math.floor(m / (t + 1)) + 1):
        t_num = pow(-1, i) * cal_factorial(m - i * (t + 1) + 1, m) / cal_factorial(1, i)
        for j in range(0, i):
            t_num = t_num * (n - j) / (n + m - 1 - j)
        for j in range(1, i * t + 1):
            t_num = t_num / (n + m - i - j)
        sum += t_num
    return sum

def cal_r_flatten_num(m, k, c, g, r):
    si = pow(2, math.ceil(math.log2(k)) + math.ceil(r * (m - math.ceil(math.log2(k))) / g))
    portion = [0 for col in range(c)]
    t_portion = [0 for col in range(c)]
    for i in range(c):
        t = 1 + i
        portion[i] = cal_below_t2(si, c, t)
    t_portion = 0
    i = 0
    while t_portion < 0.995:
        t_portion = portion[i]
        i += 1
    return i
