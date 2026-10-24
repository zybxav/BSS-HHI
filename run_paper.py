"""Run the BSS-HHI part of one paper experiment, with no parameter overrides."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import sys
from experiments.paper import run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=['E%d'%i for i in range(1,19)])
    args=parser.parse_args()
    path=run(args.experiment)
    with path.open(encoding='utf-8',newline='') as stream:
        sys.stdout.write(stream.read())

if __name__=='__main__':main()
