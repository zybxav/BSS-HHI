# -*- coding: utf-8 -*-
import numpy as np

class SparseUserMatrices:

    def __init__(self, data, ids, d):
        self.rows = data.data
        self.domain = data.dict_size
        self.ids = ids
        self.d = d
        assert len(ids) == d * d and len(set(ids)) == len(ids)
        self.lookup = {v: index for index, v in enumerate(ids) if 0 <= v < self.domain}

def projection(row, lookup, d, left, right):
    positions = sorted({lookup[v] for v in row if v in lookup})
    result = np.zeros((left.shape[1], right.shape[0]))
    for pos in positions:
        rr, cc = divmod(pos, d)
        result += np.outer(left[rr, :], right[:, cc])
    return result

def sparse_svd(self, S0, users, iterations=5, epsilon=5.0, mode='sum', theta=0.9):
    S = S0.copy()
    _, s, _ = np.linalg.svd(S, full_matrices=False)
    values = s if mode == 'sum' else s ** 2
    total = sum(values)
    partial = 0.0
    rank = 0
    for value in values:
        rank += 1
        partial += value
        if partial > total * theta:
            break
    self._chosen_rank = rank
    splits = np.array_split(np.arange(len(users.rows)), iterations)
    for indices in splits:
        if not len(indices):
            continue
        left, _, right = np.linalg.svd(S, full_matrices=False)
        left = np.asarray(left[:, :rank])
        right = np.asarray(right[:rank, :])
        flat = np.vstack([projection(users.rows[index], users.lookup, users.d, left, right).flatten() for index in indices])
        perturbed = self.l2_ball_mechanism(vectors=flat, r=1, alpha=epsilon)
        avg = np.mean(perturbed, axis=0).reshape(rank, rank)
        S = np.mat(left) @ np.mat(avg) @ np.mat(right)
    return np.asarray(S)
