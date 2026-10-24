# -*- coding: utf-8 -*-
"""Relation mining and federated HHI used by E5 and E6."""
from collections import Counter
import math
import numpy as np
from .data import Data
from .metrics import MetricsItem
from .inputs import read_rows
from ._relation import RelationKernel
from ._federated import FederatedKernel
from ._case_svim import SVIM
from ._case_refinement import SparseUserMatrices, sparse_svd

def load(name, domain, users):
    rows = read_rows('case_studies/' + name + '.txt')
    if len(rows) != users:
        raise ValueError('%s requires exactly %d users' % (name, users))
    data = Data(rows, name, domain)
    np.random.shuffle(data.data)
    return data

def run_party(task, party, data=None):
    data = data if data is not None else load(party['name'], party['domain'], party['users'])
    kernel = RelationKernel(data) if task['dataset'] == 'ifttt' else FederatedKernel(data)
    group1, group2, group3 = kernel.split_data_by_ratio(data, task['ratios'])
    matrix, ids, _ = kernel.init_matrix(group1, task['epsilon'], topk=party['m'], b=task['b'])
    users = SparseUserMatrices(group2, ids, party['m'])
    refined = sparse_svd(kernel, matrix, users, iterations=task['iterations'], epsilon=task['epsilon'], theta=task['theta'])
    candidates = kernel.get_topd_original_indices(refined, ids, int(math.ceil(task['retention'] * party['m'] ** 2)))
    group3, count = kernel.remove_non_top_items(group3, candidates)
    items, frequencies = SVIM(group3, top_k=task['local_topk'], epsilon=task['epsilon']).find_singleton_origin(count)
    items = kernel.recover_items_from_top_list(items, candidates)
    return SVIM(group3).convert_result2dict(items, frequencies)

def evaluate(task):
    relation = task['dataset'] == 'ifttt'
    truth = load(task['truth_file'][:-4], 125316 if relation else 530000, 100000 if relation else 600000)
    predictions = [run_party(task, p, truth if relation else None) for p in task['parties']]
    if relation:
        result = predictions[0]
    else:
        result = dict((Counter(predictions[1]) + Counter(predictions[0]) + Counter(predictions[2])).most_common(task['k']))
    metrics = MetricsItem(truth, task['k'])
    return dict(f1=metrics.item_f1(result), ncr=metrics.item_ncr(result), mcr=metrics.cal_mcr(result))
