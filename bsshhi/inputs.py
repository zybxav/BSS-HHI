# -*- coding: utf-8 -*-
"""Plain-text inputs; sampling uses the ordinary random state."""
import os
import gzip
import random
from functools import lru_cache
from pathlib import Path
from .algorithm import DATASET_META
from .data import Data
DATASETS = tuple((name for name in DATASET_META if name != 'ifttt'))

def data_directory():
    return Path(os.environ.get('BSSHHI_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data')))

@lru_cache(maxsize=24)
def read_rows(relative):
    path = data_directory() / relative
    if not path.is_file():
        path = path.with_suffix(path.suffix + '.gz')
    if not path.is_file():
        raise FileNotFoundError('Missing input: %s. See README.md for the input layout.' % path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        return [[int(value) for value in line.split()] for line in stream if line.strip()]

def dataset_metadata(name):
    if name not in DATASET_META:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Use a dataset name without directory separators')
        rows = read_rows('main/' + name + '.txt')
        if not rows or any(not row for row in rows):
            raise ValueError('Each input row must contain at least one item')
        DATASET_META[name] = dict(domain_size=max(2,max(max(row) for row in rows)),user_total=len(rows))
    return DATASET_META[name]

def load_dataset(name, users, relative=None):
    meta = dataset_metadata(name)
    relative = relative or 'main/' + ('UserBehavior1300' if name == 'userbehavior1300' else name) + '.txt'
    rows = read_rows(relative)
    if not 100 <= users <= len(rows):
        raise ValueError('users must be between 100 and %d' % len(rows))
    selected = random.sample(rows, users) if users < len(rows) else list(rows)
    if any(not row or any(item < 0 or item > meta['domain_size'] for item in row) for row in selected):
        raise ValueError('Input IDs must lie within the dataset domain')
    return Data(selected, name, meta['domain_size'])
