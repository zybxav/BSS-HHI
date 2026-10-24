"""Measure the BSS-HHI series of E1--E18 with fixed paper settings."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from bsshhi.algorithm import BSSHHI,DATASET_META
from bsshhi.data import Data
from bsshhi.inputs import load_dataset,read_rows
from bsshhi.metrics import MetricsItem
from bsshhi.protocol import reconstruct,refine,retain,estimate,scores
from bsshhi import case_studies,domain_encoding,single_value

REPEATS=10
ROOT=Path(__file__).resolve().parents[1]

def configurations(experiment):
    return json.loads((Path(__file__).parent/'configs'/(experiment+'.json')).read_text(encoding='utf-8'))

def singular_values(matrix,theta):
    values=np.linalg.svd(matrix,compute_uv=False)
    total=float(values.sum());energy=values**2;energy_total=float(energy.sum())
    cumulative=np.cumsum(energy)/energy_total if energy_total else np.zeros_like(values)
    chosen=min(len(values),int(np.searchsorted(np.cumsum(values),theta*total,side='right'))+1)
    probability=values/total if total else np.zeros_like(values)
    positive=probability[probability>0]
    return dict(matrix_side=matrix.shape[0],top1_energy=float(cumulative[0]),
        top5_energy=float(cumulative[min(4,len(values)-1)]),
        rank90=int(np.searchsorted(cumulative,.9)+1) if energy_total else 0,
        effective_rank=float(np.exp(-np.sum(positive*np.log(positive)))) if total else 0.,
        chosen_rank=chosen,energy_at_chosen_rank=float(cumulative[chosen-1]))

def standard(experiment,task,data):
    parameters=task['parameters'];epsilon=task['epsilon'];topk=task['topk']
    bss=BSSHHI(data)
    group1,group2,group3=bss.split_data_by_ratio(data,task['cohorts'])
    x1,union,matrix,padded=reconstruct(bss,group1,epsilon,parameters,task['allocation'])
    if experiment=='E18':
        rows=[dict(ordering='score',**singular_values(matrix,parameters['theta']))]
        if epsilon==4:
            shuffled=np.random.permutation(matrix.ravel()).reshape(matrix.shape)
            rows.append(dict(ordering='random',**singular_values(shuffled,parameters['theta'])))
        return rows
    measured={}
    if experiment=='E12':measured['chosen_rank']=singular_values(matrix,parameters['theta'])['chosen_rank']
    skip=task.get('variant','').startswith('no_svd')
    refined=refine(bss,group2,matrix,padded,epsilon,parameters,skip=skip)
    candidates=retain(bss,refined,padded,len(union),topk,parameters['rho2'],data.dict_size)
    predictions=estimate(bss,group3,candidates,epsilon,topk)
    if experiment=='E7':
        truth={item[0] for item in MetricsItem(data,topk).true_item_dict}
        denominator=max(1,len(truth))
        return [dict(phase1_recall=len(set(x1)&truth)/denominator,
            phase12_recall=len(set(union)&truth)/denominator,
            phase3_recall=len(set(candidates)&truth)/denominator,
            final_recall=len({item[0] for item in predictions}&truth)/denominator)]
    result=scores(data,predictions,topk)
    if experiment in ('E4','E9','E10','E11','E12','E13','E14','E15'):result={'f1':result['f1']}
    result.update(measured)
    return [result]

def evaluate(experiment,task,repeat,shared):
    if experiment in ('E5','E6'):return [case_studies.evaluate(task)]
    if experiment=='E8':
        rows=read_rows(task['input']);users=task['n']
        if users>len(rows):raise ValueError('Insufficient users for E8')
        if any(len(row)!=1 for row in rows):raise ValueError('E8 input must contain one value per user')
        values=np.array([row[0] for row in rows],dtype=np.int64)
        if len(values)>users:values=np.random.choice(values,users,replace=False)
        return [single_value.evaluate(values,DATASET_META[task['dataset']]['domain_size'],task['epsilon'],task['topk'],task['parameters'],task['dataset'])]
    key=(task['dataset'],repeat)
    if experiment in ('E16','E17'):
        if key not in shared:
            data=load_dataset(task['dataset'],task['n'],task['input'])
            shared[key]=(data,np.random.permutation(np.arange(1,data.dict_size+1)))
        data,order=shared[key]
        if task['permutation_fraction']>0:
            mapping=domain_encoding.partial_permutation(data.dict_size,task['permutation_fraction'],order)
            return [domain_encoding.evaluate(data,task['epsilon'],task['topk'],task['encoding_parameters'],mapping)]
        return [{'f1':standard(experiment,task,data)[0]['f1']}]
    if experiment=='E14':
        key=(task['dataset'],task['input'],repeat)
        if key not in shared:
            rows=read_rows(task['input']);order=np.random.permutation(len(rows));shared[key]=[rows[i] for i in order]
        rows=shared[key]
        if task['n']>len(rows):raise ValueError('Insufficient users for E14')
        data=Data(rows[:task['n']],task['dataset'],DATASET_META[task['dataset']]['domain_size'])
    else:data=load_dataset(task['dataset'],task['n'],task['input'])
    return standard(experiment,task,data)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)

def run(experiment):
    tasks=configurations(experiment);shared={};summary=[]
    for task in tasks:
        groups={}
        for repeat in range(REPEATS):
            for measured in evaluate(experiment,task,repeat,shared):
                measured=dict(measured);ordering=measured.pop('ordering',None)
                if not all(np.isfinite(value) for value in measured.values()):raise ValueError('Non-finite paper metric')
                groups.setdefault(ordering,[]).append(measured)
        labels={key:task[key] for key in ('dataset','epsilon','topk','n','factor','level','variant','permutation_fraction') if key in task}
        if experiment in ('E5','E6'):
            labels.update(topk=task['k'],n=sum(party['users'] for party in task['parties']))
        for ordering,records in groups.items():
            row=dict(labels)
            if ordering is not None:row['ordering']=ordering
            for metric in records[0]:
                values=[record[metric] for record in records]
                row[metric+'_mean']=statistics.mean(values)
                row[metric+'_std']=statistics.stdev(values) if len(values)>1 else 0.
            summary.append(row)
    target=ROOT/'results'/(experiment+'.csv');write_csv(target,summary)
    if experiment=='E15':
        profile=[]
        for task in tasks:
            if task['epsilon']!=1:continue
            counts=Counter(item for row in read_rows(task['input']) for item in row)
            values=sorted(counts.values(),reverse=True)[:64]
            for rank,count in enumerate(values,start=1):profile.append(dict(alpha=task['level'],rank=rank,relative_frequency=count/values[0]))
        write_csv(ROOT/'results/E15_profile.csv',profile)
    return target
