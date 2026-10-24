"""Ordered item-coordinate interface for the existing SVD-FO implementation."""
import math
import numpy as np

from .data_rm import Data
from .svd_fo import LDP_RM


class OrderedSVD(LDP_RM):
    """Keep row and column IDs distinct when using the relation-based engine."""

    def __init__(self, data, epsilon, topk, side, domain):
        super().__init__(data, epsilon, min(64, 2 * side), topk, topk)
        self.side = side
        self.original_domain = domain
        self.pair_scores = {}

    def domain2(self, U, V, length_limit, clip_coefficient=1.0):
        # Ordered-coordinate matrices include diagonal cells. They are valid
        # item IDs and must be included in the projection bounds.
        U, V = np.asarray(U), np.asarray(V)
        limit = max(1, min(U.shape[0] * V.shape[1], int(length_limit) ** 2))
        bounds = np.empty(U.shape[1], dtype=object)
        for i in range(U.shape[1]):
            coefficients = np.sort(np.outer(U[:, i], V[i]).ravel())
            lower = np.minimum(coefficients[:limit], 0).sum()
            upper = np.maximum(coefficients[-limit:], 0).sum()
            bounds[i] = (float(upper * clip_coefficient), float(lower * clip_coefficient))
        return bounds

    def find_matrix_ks(self, matrix, singletons_row, singletons_col):
        candidates = []
        for i, row in enumerate(singletons_row):
            for j, column in enumerate(singletons_col):
                item = (int(row) - 1) * self.side + int(column) - self.side
                value = float(matrix[i, j])
                if 1 <= item <= self.original_domain and np.isfinite(value):
                    candidates.append((value, item, (int(row), int(column))))
        candidates.sort(key=lambda entry: (-entry[0], entry[1]))
        self.pair_scores = {pair: value for value, item, pair in candidates[:self.top_ks]}
        return list(self.pair_scores), dict(self.pair_scores)


def identify(data, epsilon, topk):
    """Run the existing SVIM/iterative-SVD stages on ordered item coordinates."""
    side = int(math.ceil(math.sqrt(data.dict_size)))
    pairs = [[((item - 1) // side + 1, side + (item - 1) % side + 1)
              for item in row] for row in data.data]
    pair_data = Data(pairs, data.data_name, 2 * side)
    protocol = OrderedSVD(pair_data, epsilon, topk, side, data.dict_size)
    split = int(.5 * len(pairs))
    svd_end = int(.99 * len(pairs))
    items, frequencies, length_limit = protocol.svsm.svim.find_singleton(split)
    row_entries = [(int(item), float(value)) for item, value in zip(items, frequencies)
                   if 1 <= item <= side]
    column_entries = [(int(item), float(value)) for item, value in zip(items, frequencies)
                      if side < item <= 2 * side]
    size = min(len(row_entries), len(column_entries))
    if not size:
        return {}
    rows, row_frequencies = zip(*row_entries[:size])
    columns, column_frequencies = zip(*column_entries[:size])
    protocol.find_itemset_svd_origin(
        list(rows), list(columns), list(row_frequencies), list(column_frequencies),
        split, svd_end, max(1, int(length_limit)), sing_num=0,
        method='AMN', use_group=True, group_num=5, test='test_constant', task='FISM')
    return {((row - 1) * side + column - side,): value
            for (row, column), value in protocol.pair_scores.items()}
