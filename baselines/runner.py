"""Shared dataset, method and metric interfaces for the comparison methods."""
import contextlib
import io
import math
import random
import secrets
import time

import numpy as np

from bsshhi.data import Data
from bsshhi.inputs import dataset_metadata, load_dataset, read_rows
from bsshhi.metrics import MetricsItem


SET_METHODS = ('svim', 'svdfo', 'pemset')
HHI_METHODS = ('treehist', 'bitstogram', 'privateexpander', 'hadaheavy')
FO_METHODS = ('olh', 'oue', 'sue', 'cms', 'hcms', 'hadaoracle', 'pgr', 'mss', 'pan-ocms', 'hashtogram')
METHODS = SET_METHODS + HHI_METHODS + FO_METHODS


def load_input(dataset, users, single=False):
    """Use the provided E8 input for single-valued methods when available."""
    metadata = dataset_metadata(dataset)
    if not single:
        loaded = load_dataset(dataset, users)
        rows = [list(dict.fromkeys(row)) for row in loaded.data]
        input_kind = 'set-valued'
    else:
        filename = 'UserBehavior1300' if dataset == 'userbehavior1300' else dataset
        try:
            stored = read_rows('single/' + filename + '.single.n100000.txt')
        except FileNotFoundError:
            stored = None
        if stored is not None and users <= len(stored):
            if any(len(row) != 1 for row in stored):
                raise ValueError('Single-valued inputs must contain one item per row')
            selected = random.sample(stored, users) if users < len(stored) else list(stored)
            rows = [list(row) for row in selected]
        else:
            loaded = load_dataset(dataset, users)
            rows = [[random.choice(list(dict.fromkeys(row)))] for row in loaded.data]
        input_kind = 'single-valued'
    domain = int(metadata['domain_size'])
    if any(not row or any(item < 1 or item > domain for item in row) for row in rows):
        raise ValueError('Comparison methods require item IDs in [1, domain_size]')
    return Data(rows, dataset, domain), input_kind


def _single_factory(method):
    if method == 'treehist':
        from .hhi.treehist_paper import TreeHistPaper
        return TreeHistPaper()
    if method == 'bitstogram':
        from .hhi.bitstogram_core import BitstogramCore
        return BitstogramCore()
    if method == 'privateexpander':
        from .hhi.private_expander_sketch_core import PrivateExpanderSketchCore
        return PrivateExpanderSketchCore()
    if method == 'hadaheavy':
        from .hhi.hadaheavy_core import HadaHeavyCore
        return HadaHeavyCore()
    if method in ('olh', 'oue', 'sue'):
        from .fo.standard import OLHOracle, UnaryEncodingOracle
        return OLHOracle() if method == 'olh' else UnaryEncodingOracle(method == 'oue')
    if method in ('cms', 'hcms'):
        from .fo.apple_cms import AppleCMS
        return AppleCMS(hadamard=method == 'hcms')
    if method == 'hadaoracle':
        from .fo.hadaoracle import HadaOracle
        return HadaOracle()
    if method == 'pgr':
        from .fo.pgr import PGROracle
        return PGROracle()
    if method == 'mss':
        from .fo.mss import MSSOracle
        return MSSOracle()
    if method == 'pan-ocms':
        from .fo.pan_ocms import PanOCMS
        return PanOCMS()
    if method == 'hashtogram':
        from .fo.hashtogram import Hashtogram
        return Hashtogram()
    raise ValueError('Unknown method: ' + method)


def _pemset(data, epsilon, topk):
    from .main.PEMSET.method import data_to_prefix, PEM_wheel_flatten
    # The paper comparison uses sets of size 10 and MWheel with six layers.
    size = 10
    padding = range(data.dict_size + 1, data.dict_size + 10000)
    candidate_size = 1 << int(math.ceil(math.log2(max(2, topk))))
    bits = 64
    layers = 6
    rows = []
    for original in data.data:
        values = random.sample(original, size) if len(original) >= size else list(original) + random.sample(padding, size - len(original))
        rows.append([int(item) for item in values])
    prefixes, schedule = data_to_prefix(rows, len(rows), layers, size, bits, candidate_size)
    items, estimates = PEM_wheel_flatten(prefixes, candidate_size, bits, layers, size, epsilon, len(rows), schedule)
    # Match the author's evaluation: padding/out-of-domain candidates remain
    # in the returned answer and count as false positives.
    predictions = [(int(item), float(value) * len(rows)) for item, value in zip(items, estimates)]
    return {(item,): value for item, value in predictions[:topk]}


def run_method(method, data, epsilon, topk, input_kind):
    started = time.perf_counter()
    implementation = method
    # Legacy methods print intermediate matrices; only final results belong
    # in the command-line output.
    with contextlib.redirect_stdout(io.StringIO()):
        if method == 'svim':
            from bsshhi.svim import SVIM
            items, estimates = SVIM(data, top_k=topk, epsilon=epsilon).find_singleton_origin(len(data.data))
            predictions = {(int(item),): float(value) for item, value in zip(items, estimates)
                           if 1 <= item <= data.dict_size and np.isfinite(value)}
            implementation = 'SVIM'
        elif method == 'svdfo':
            from .main.SVD_FO.adapter import identify
            predictions = identify(data, epsilon, topk)
            implementation = 'SVD-FO (ordered-coordinate adaptation)'
        elif method == 'pemset':
            predictions = _pemset(data, epsilon, topk)
            implementation = 'PEMSET / MWheel'
        else:
            protocol = _single_factory(method)
            implementation = protocol.name
            values = np.asarray([row[0] - 1 for row in data.data], dtype=np.int64)
            # Internal author interfaces need a random key. It is generated
            # afresh, without requiring a user-specified seed.
            random_key = secrets.randbits(32)
            if method in HHI_METHODS:
                result = protocol.run(values, data.dict_size, epsilon, topk, random_key)
                predictions = {(int(item),): float(value) for item, value in
                               zip(result['items'], result['estimated_frequencies'])}
            else:
                estimates = np.asarray(protocol.estimate(values, data.dict_size, epsilon, random_key), dtype=float)
                if estimates.shape != (data.dict_size,) or not np.all(np.isfinite(estimates)):
                    raise ValueError('The frequency oracle did not return finite full-domain estimates')
                ids = np.arange(1, data.dict_size + 1)
                order = np.lexsort((ids, -estimates))[:topk]
                predictions = {(int(ids[i]),): float(estimates[i]) for i in order}
    if len(predictions) > topk or any((method != 'pemset' and not 1 <= item[0] <= data.dict_size) or not np.isfinite(value)
                                    for item, value in predictions.items()):
        raise ValueError('Method returned invalid item IDs, estimates, or output cardinality')
    metric = MetricsItem(data, topk)
    return dict(method=method, implementation=implementation, dataset=data.data_name,
                input_type=input_kind, epsilon=epsilon, top_k=topk, users=len(data.data),
                f1=metric.item_f1(predictions), ncr=metric.item_ncr(predictions),
                mcr=metric.cal_mcr(predictions), seconds=time.perf_counter() - started)


def run(methods, dataset, epsilon, topk, users):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    if users < 100:
        raise ValueError('users must be at least 100')
    domain = int(dataset_metadata(dataset)['domain_size'])
    if not 1 <= topk <= domain:
        raise ValueError('top-k is outside the domain')
    inputs, output = {}, []
    for method in methods:
        if method not in METHODS:
            raise ValueError('Unknown method: ' + method)
        single = method not in SET_METHODS
        if single not in inputs:
            inputs[single] = load_input(dataset, users, single)
        data, kind = inputs[single]
        output.append(run_method(method, data, epsilon, topk, kind))
    return output
