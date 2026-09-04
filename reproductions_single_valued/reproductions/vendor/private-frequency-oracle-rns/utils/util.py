# This file includes utility functions adapted from the PGR C++ implementation:
# https://github.com/minilek/private_frequency_oracles/
#
# BSD 3-Clause License

# Copyright (c) 2022, Jelani Nelson
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import random

def is_prime(q):
    if q < 2:
        return False
    if q == 2:
        return True
    if q % 2 == 0:
        return False
    for d in range(3, int(q ** 0.5) + 1, 2):
        if q % d == 0:
            return False
    return True

def fast_mod_pow(x, n, q):
    if n == 0:
        return 1
    z = fast_mod_pow(x, n // 2, q)
    z = (z * z) % q
    if n % 2 == 1:
        z = (z * x) % q
    return z

def compute_inverse_table(q):
    assert is_prime(q)
    inv = [0] * q
    for i in range(1, q):
        for j in range(1, q):
            if (i * j) % q == 1:
                inv[i] = j
                break
    return inv

def first_non_zero(v):
    for i, val in enumerate(v):
        if val != 0:
            return i
    return -1

def canonicalize(v, q, qinv):
    i = first_non_zero(v)
    assert i != -1
    g = qinv[v[i]]
    for j in range(len(v)):
        v[j] = (v[j] * g) % q

def canonicalized_index_to_vec(j, l, q):
    tot = 0
    qpow = 1
    num_zeroes = l - 1
    while j >= tot + qpow:
        tot += qpow
        qpow *= q
        num_zeroes -= 1
    ret = [0] * l
    ret[num_zeroes] = 1
    r = j - tot
    at = l - 1
    while r > 0:
        ret[at] = r % q
        r //= q
        at -= 1
    return ret

def canonicalized_vec_to_index(v, q, qpows):
    leftmost_one = first_non_zero(v)
    assert leftmost_one != -1 and v[leftmost_one] == 1
    ret = (qpows[len(v) - 1 - leftmost_one] - 1) // (q - 1)
    c = 0
    for i in range(leftmost_one + 1, len(v)):
        c = c * q + v[i]
    return ret + c

def randvec_with_specified_dot_product(v, s, q, qinv, rng=random):
    while True:
        ret = [rng.randint(0, q - 1) for _ in range(len(v))]
        i = first_non_zero(v)
        assert i != -1
        z = sum(v[j] * ret[j] for j in range(len(v)) if j != i) % q
        ret[i] = ((q + s - z) * qinv[v[i]]) % q
        if sum(ret) != 0:
            return ret

def mod_dot_product(u, v, q):
    return sum((u[i] * v[i]) % q for i in range(len(u))) % q


def enumerate_vectors_of_dim(t, q):
    """
    Yield all q-ary vectors of dimension t (i.e., elements of 𝔽_q^t).
    """
    if t <= 0:
        yield []
    else:
        for tail in enumerate_vectors_of_dim(t - 1, q):
            for i in range(q):
                yield tail + [i]


def all_orthogonal_vectors(v, q, qinv):
    """
    Yield every canonical vector u (in 𝔽_q^t, u[0] = 1 or u = (0,…,0))
    such that ⟨u , v⟩ = 0  in 𝔽_q.
    """
    t = len(v)
    for raw in enumerate_vectors_of_dim(t, q):         # any order is fine
        if mod_dot_product(raw, v, q) == 0:
            u = raw[:]
            canonicalize(u, q, qinv)
            yield u
