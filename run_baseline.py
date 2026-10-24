"""Run the comparison methods with dataset, epsilon, top-k and user settings."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json
from baselines.runner import METHODS, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', nargs='+', choices=METHODS, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--top-k', type=int, default=32)
    parser.add_argument('--users', type=int, default=100000)
    args = parser.parse_args()
    try:
        results = run(args.method, args.dataset, args.epsilon, args.top_k, args.users)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
