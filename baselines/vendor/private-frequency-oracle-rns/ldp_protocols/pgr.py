# Projective Geometry Response (PGR) Protocol — Python Implementation
#
# This implementation is based on the original C++ version by Jelani Nelson:
#     https://github.com/minilek/private_frequency_oracles/
# but has been adapted and extended to integrate with our BaseProtocol framework.
#
# In addition to translating core components, we introduce the following modifs:
#   • a unified interface consistent with our protocol benchmark infrastructure,
#   • analytical and empirical attackability (DRA/ASR) routines.
#
# The original code is licensed under the BSD 3-Clause License, reproduced below.
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

from __future__ import annotations

import random
import numpy as np
from typing import List, Optional
from utils.message import Message
from utils import util
from ldp_protocols.base import BaseProtocol
import sympy as sp   

class ProjectiveGeometryResponse(BaseProtocol):
    def __init__(self, k: int, epsilon: float, q: Optional[int] = None, debug=False, seed=None):
        super().__init__(k, epsilon)
        
        self.debug = debug
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        # Auto-select q and t if not provided
        if q is None:
            self.q, self.t = self._choose_q_t(k, epsilon)
        else:
            self.q = q
            self.t = 2
            while (self.q**self.t - 1) // (self.q - 1) < k:
                self.t += 1

        # Build qpows
        self.qpows = [1, self.q]
        qpow = self.q
        for _ in range(2, self.t + 1):
            qpow *= self.q
            self.qpows.append(qpow)

        self.k_original = k
        self.k = (qpow - 1) // (self.q - 1)
        self.qinv = util.compute_inverse_table(self.q)
        self.cint = (self.qpows[self.t - 2] - 1) // (self.q - 1) if self.t >= 2 else 0
        self.cset = (self.qpows[self.t - 1] - 1) // (self.q - 1)

        self.p = 1.0 / ((np.exp(self.epsilon) - 1) * self.cset + self.k)
        s = np.exp(self.epsilon) * self.p * self.cset
        r = self.p * ((np.exp(self.epsilon) - 1) * self.cint + self.cset)
        self.alpha = 1.0 / (s - r)
        self.beta = -self.alpha * r

        if self.debug:
            print(f"PGR: k_original={self.k_original}, k={self.k}, q={self.q}, t={self.t}")
            print(f"     cint={self.cint}, cset={self.cset}, p={self.p:.6f}")
            print(f"     alpha={self.alpha:.6f}, beta={self.beta:.6f}")

    def obfuscate(self, data: np.ndarray) -> List[Message]:
        """Apply local randomization to input data"""
        return [self.local_randomizer(int(x)) for x in data]

    def estimate_raw(self, reports: List[Message]) -> np.ndarray:
        """Get raw frequency estimates (before normalization)"""
        est = self.estimate_all_freqs(reports)
        return np.array(est[:self.k_original])
        
    def estimate(self, reports: List[Message]) -> np.ndarray:
        """Get normalized frequency estimates"""
        est = np.maximum(self.estimate_raw(reports), 0.0)
        return est / np.sum(est)
    
    def get_mse(self, n: int) -> float:
        """Theoretical MSE bound"""
        return (4 * np.exp(self.epsilon)) / (n * (np.exp(self.epsilon) - 1) ** 2)
    

    def _message_belongs_to_S(self, y: int, x: int) -> bool:
        """
        Return True if the PGR message `y` belongs to the preferred set S(x).
        Equivalent to: mod_dot_product(vec(y), vec(x)) == 0.
        """
        v_x = util.canonicalized_index_to_vec(x, self.t, self.q)
        v_y = util.canonicalized_index_to_vec(y, self.t, self.q)
        return util.mod_dot_product(v_x, v_y, self.q) == 0
    
    def attack(self, report) -> int:
        y = report.read()

        candidates = [
            x for x in range(self.k_original)
            if self._message_belongs_to_S(y, x)
        ]

        if len(candidates) == 0:
            return self.rng.randint(0, self.k_original - 1)

        return self.rng.choice(candidates)

    def get_asr(self) -> float:
        """
        Exact Bayes-optimal single-message ASR for PGR under a uniform prior
        on x in {0, ..., k_original-1}, even when k_original < self.k.
        """
        eps = self.epsilon
        exp_eps = np.exp(eps)

        K0    = self.k_original   # prior support: 0..K0-1
        K_msg = self.k            # number of possible messages
        c_set = self.cset

        # Mechanism probabilities used in local_randomizer
        Z = (exp_eps - 1) * c_set + K_msg
        p = 1.0 / Z
        prob_pref    = exp_eps * p   # P(y|x) when y in S(x)
        prob_nonpref = p             # P(y|x) otherwise

        # ASR = (1/K0) * sum_y max_x P(y|x)
        count_with_pref = 0

        for y in range(K_msg):
            # Does this message have at least one active input x<K0 that prefers it?
            if any(self._message_belongs_to_S(y, x) for x in range(K0)):
                count_with_pref += 1

        count_without_pref = K_msg - count_with_pref

        asr = (1.0 / K0) * (
            count_with_pref   * prob_pref +
            count_without_pref * prob_nonpref
        )
        return float(asr)

    # def get_asr(self) -> float:
    #     """
    #     Closed-form ASR for PGR when the prior uses the full projective domain:
    #     k_original == k == (q^t - 1)/(q - 1).
    #     """

    #     exp_eps = np.exp(self.epsilon)
    #     K       = self.k
    #     c_set   = self.cset

    #     return float(exp_eps / (K + (exp_eps - 1) * c_set))

    @staticmethod
    def _choose_q_t(k, eps):
        # Match the authors' official C++ implementation:
        # q is the smallest prime not below ceil(exp(epsilon) + 1).
        q_lower_bound = max(2, int(np.ceil(np.exp(eps) + 1)))
        q = (
            q_lower_bound
            if sp.isprime(q_lower_bound)
            else int(sp.nextprime(q_lower_bound))
        )

        t = 2
        while (q**t - 1) // (q - 1) < k:
            t += 1
        return q, t

    def local_randomizer(self, x: int) -> Message:
        """Apply local differential privacy to input x"""
        # Ensure x is within valid range
        if x >= self.k:
            x = x % self.k
            
        # Convert to canonical vector
        v = util.canonicalized_index_to_vec(x, self.t, self.q)
        
        # Apply randomization
        if self.rng.random() <= self.p * np.exp(self.epsilon) * self.cset:
            # Output random vector orthogonal to v (dot product = 0)
            ret = util.randvec_with_specified_dot_product(v, 0, self.q, self.qinv, self.rng)
        else:
            # Output random vector with non-zero dot product
            s = self.rng.randint(1, self.q - 1)
            ret = util.randvec_with_specified_dot_product(v, s, self.q, self.qinv, self.rng)
        
        # Canonicalize the result
        util.canonicalize(ret, self.q, self.qinv)
        
        # Convert back to index
        result_index = util.canonicalized_vec_to_index(ret, self.q, self.qpows)
        return Message(result_index)

    def estimate_freq(self, x: int, messages: List[Message]) -> float:
        """Estimate frequency of a single element x"""
        # Ensure x is within valid range
        if x >= self.k:
            x = x % self.k
            
        v = util.canonicalized_index_to_vec(x, self.t, self.q)
        count = 0
        
        for m in messages:
            msg_index = m.read()
            # Ensure message index is valid
            if msg_index < self.k:
                u = util.canonicalized_index_to_vec(msg_index, self.t, self.q)
                if util.mod_dot_product(u, v, self.q) == 0:
                    count += 1
        
        return self.alpha * count + self.beta * len(messages)

    def estimate_all_freqs(self, messages: List[Message]) -> List[float]:
        """Estimate frequencies for all elements"""
        # Initialize histogram
        n = len(messages)
        y = [0] * self.k

        for m in messages:
            msg_index = m.read()
            if 0 <= msg_index < self.k:
                y[msg_index] += 1

        if self.t <= 3:
            ret = []
            for x in range(self.k):
                v = util.canonicalized_index_to_vec(x, self.t, self.q)
                last_nz = util.first_non_zero(v[::-1])
                last_nz = self.t - 1 - last_nz if last_nz != -1 else -1

                if last_nz == -1:
                    freq = self.beta
                else:
                    count = 0
                    offset = 0
                    for j in range(self.t - 1, -1, -1):
                        count += self.count_orthogonal_messages(v, v[j], j + 1, offset, last_nz, y)
                        offset += self.qpows[self.t - 1 - j]
                    freq = (self.alpha * count + self.beta * n) / n

                ret.append(freq)
            return ret
        else:
            T = self.dp_bottom_up(y)
            return [(self.alpha * T[i] + self.beta * n) / n for i in range(self.k)]

    def count_orthogonal_messages(self, v, dp_so_far, at, index, last_nz, y):
        """Recursively count orthogonal messages"""
        if at == len(v):
            return 0 if dp_so_far else y[index] if index < len(y) else 0
        elif at != last_nz:
            total = 0
            for d in range(self.q):
                new_dp = (dp_so_far + d * v[at]) % self.q
                new_index = index + d * self.qpows[self.t - 1 - at]
                if new_index < len(y):  # Bounds check
                    total += self.count_orthogonal_messages(v, new_dp, at + 1, new_index, last_nz, y)
            return total
        else:
            # Solve for d such that dp_so_far + d * v[last_nz] = 0 (mod q)
            if v[last_nz] != 0:  # Additional safety check
                d = ((self.q - dp_so_far) * self.qinv[v[last_nz]]) % self.q
                new_dp = (dp_so_far + d * v[at]) % self.q
                new_index = index + d * self.qpows[self.t - 1 - at]
                if new_index < len(y):  # Bounds check
                    return self.count_orthogonal_messages(v, new_dp, at + 1, new_index, last_nz, y)
            return 0

    def dp_bottom_up(self, y: List[int]) -> List[int]:
        """
        Space-optimised dynamic programme that computes
            T[u] = Σ_{v∈S_u} y[v]
        """
        q, t, K     = self.q, self.t, self.k
        qpows, qinv = self.qpows, self.qinv
        assert len(y) == K

        # ────────────────────────────── workspace size
        N = K + 1                                # length-0 layer (z≠0 dropped)
        for l in range(1, t):
            N = max(N,
                     ((qpows[l]      - 1)//(q - 1) + 1) *
                     ((qpows[t - l]  - 1)//(q - 1) + 1) * q)

        last  = [0]*N    # DP table for length+1
        next_ = [0]*N    # DP table for current length

        # ───────────────────────────── base layer  (length == t)
        for a in range(1, K + 1):
            last[a] = y[a - 1]                   # only z = 0 entries are kept
        lastA, lastB = K + 1, 1                  # |a| and |b| in ‘last’

        # Helper
        def decompose(vindex: int, l: int) -> tuple[int, int, int]:
            if vindex == 0:                      # the all-zero suffix
                return 0, 1, 0                   # (v[0], g, idx)
            v     = util.canonicalized_index_to_vec(vindex - 1, l, q)
            v0    = v[0]
            vsuff = v[1:]
            g     = next((x for x in vsuff if x != 0), 1)
            if any(vsuff):
                util.canonicalize(vsuff, q, qinv)
                vsuff_idx = util.canonicalized_vec_to_index(vsuff, q, qpows) + 1
            else:
                vsuff_idx = 0
            return v0, g, vsuff_idx

        # ───────────────────────────── bottom-up sweep
        for length in range(t - 1, -1, -1):
            curA = (qpows[length]     - 1)//(q - 1) + 1
            curB = (qpows[t - length] - 1)//(q - 1) + 1

            # zero the working buffer
            next_[:curB * curA * q if length else curB] = [0] * \
                (curB * curA * q if length else curB)

            for b in range(curB):
                vb0, g, vbsuff_idx = decompose(b, t - length)
                ginv = qinv[g]

                for a in range(curA):
                    if length == 0:              # only z = 0 matters
                        calc  = last[vbsuff_idx * lastA * q + 0]   # append 0
                        calc += last[vbsuff_idx * lastA * q + q +
                                     ((q - vb0) * ginv) % q]       # append 1
                        next_[b] = calc
                    else:
                        extension = 0 if a == 0 else 2 + (a - 1) * q
                        for z in range(q):
                            calc = 0
                            max_d = 1 if a == 0 else q - 1
                            for d in range(max_d + 1):
                                new_dp = (((q + z - vb0 * d) % q) * ginv) % q
                                if length == t - 1:                # first step
                                    # last only kept z = 0 here
                                    calc += 0 if new_dp else last[extension + d]
                                else:
                                    idx = (vbsuff_idx * lastA * q +
                                           (extension + d) * q + new_dp)
                                    calc += last[idx]
                            next_[b * curA * q + a * q + z] = calc

            last, next_ = next_, last            # roll the layers
            lastA, lastB = curA, curB

        # ───────────────────────────── extract answers
        return last[1:K + 1]                     # skip the a = 0 slot

    def get_cset(self):
        return self.cset

    def get_cint(self):
        return self.cint

    def get_t(self):
        return self.t
