"""Run BSS-HHI with dataset/task settings and the parameters in paper Table 1."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from bsshhi.inputs import dataset_metadata,load_dataset
from bsshhi.protocol import identify,scores,validate
from experiments.settings import selected

def sequence(value):
    if value.lower()=='none':return []
    try:return [int(item) for item in value.split(',')]
    except ValueError:raise argparse.ArgumentTypeError('c must be comma-separated integers, or none')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--epsilon',type=float,required=True)
    parser.add_argument('--top-k',type=int,default=32)
    parser.add_argument('--users',type=int,default=100000)
    parser.add_argument('--c',type=sequence)
    parser.add_argument('--x1-ratio',type=float)
    parser.add_argument('--c0',type=int)
    parser.add_argument('--rho2',type=float)
    parser.add_argument('--theta',type=float)
    parser.add_argument('--iterations',type=int)
    args=parser.parse_args()
    if not args.epsilon>0:parser.error('epsilon must be positive')
    domain=dataset_metadata(args.dataset)['domain_size']
    if not 1<=args.top_k<=domain:parser.error('top-k is outside the domain')
    if args.users<100:parser.error('users must be at least 100')
    setting=selected(args.dataset,args.epsilon)
    if setting is None:
        if any(getattr(args,key) is None for key in ('c','x1_ratio','c0','rho2')):
            parser.error('For a new dataset, specify --c, --x1-ratio, --c0 and --rho2')
        setting=dict(parameters=dict(c=args.c,x1=1,c0=args.c0,rho2=args.rho2,theta=.9,iterations=5),
                     allocation=[.5]*max(1,len(args.c)))
    parameters=setting['parameters']
    for name in ('c','c0','rho2','theta','iterations'):
        value=getattr(args,name)
        if value is not None:parameters[name]=value
    if args.x1_ratio is not None:
        if not 0<args.x1_ratio<=1:parser.error('x1-ratio must lie in (0,1]')
        parameters['x1']=max(1,int(round(args.x1_ratio*domain)))
    validate(parameters,domain)
    data=load_dataset(args.dataset,args.users)
    predictions=identify(data,args.epsilon,args.top_k,parameters,setting['allocation'])
    print(json.dumps(dict(dataset=args.dataset,epsilon=args.epsilon,top_k=args.top_k,users=args.users,
        **scores(data,predictions,args.top_k)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
