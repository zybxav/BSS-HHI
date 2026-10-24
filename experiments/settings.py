"""One selected setting per dataset and privacy regime."""
import copy
import json
from pathlib import Path

def selected(dataset,epsilon):
    rows=json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8')).get(dataset)
    if rows is None:return None
    eligible=[row for row in rows if row['epsilon_min']<=epsilon]
    return copy.deepcopy((eligible or rows[:1])[-1])
