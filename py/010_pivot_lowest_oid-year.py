#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:36:38 2018

@author: Kazuki

focus on highest year

keys: object_id, passband


"""

import numpy as np
import pandas as pd
import os
from glob import glob
from multiprocessing import cpu_count, Pool
from itertools import combinations
import utils

PREF = 'f010'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

def quantile(n):
    def quantile_(x):
        return np.percentile(x, n)
    quantile_.__name__ = 'q%s' % n
    return quantile_

stats = ['min', 'max', 'mean', 'median', 'std', 'kurt', quantile(25), quantile(75)]

num_aggregations = {
    'flux':        stats,
    'flux_norm1':  stats,
    'flux_norm2':  stats,
    'flux_err':    stats,
    'detected':    stats,
    }

def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    # -178 ~ date ~ +178
    idxmax = df.groupby('object_id').flux.idxmin()
    base = df.iloc[idxmax][['object_id', 'date']]
    li = [base]
    for i in range(178):
        i += 1
        lag  = base.copy()
        lead = base.copy()
        lag['date']  -= i
        lead['date'] += i
        li.append(lag)
        li.append(lead)
    
    keep = pd.concat(li)
    
    df = pd.merge(keep, df, on=['object_id', 'date'], how='inner')
    
    pt = pd.pivot_table(df, index=['object_id'], columns=['passband'], 
                        aggfunc=num_aggregations)
    
    pt.columns = pd.Index([f'pb{e[2]}_{e[0]}_{e[1]}' for e in pt.columns.tolist()])
    
    # std / mean
    col_std = [c for c in pt.columns if c.endswith('_std')]
    for c in col_std:
        pt[f'{c}-d-mean'] = pt[c]/pt[c.replace('_std', '_mean')]
    
    # max / min, max - min
    col_max = [c for c in pt.columns if c.endswith('_max')]
    for c in col_max:
        pt[f'{c}-d-min'] = pt[c]/pt[c.replace('_max', '_min')]
        pt[f'{c}-m-min'] = pt[c]-pt[c.replace('_max', '_min')]
    
    # compare passband
    col = pd.Series([f'{c[3:]}' for c in pt.columns if c.startswith('pb0')])
    for c1,c2 in list(combinations(range(6), 2)):
        col1 = (f'pb{c1}'+col).tolist()
        col2 = (f'pb{c2}'+col).tolist()
        for c1,c2 in zip(col1, col2):
            pt[f'{c1}-d-{c2}'] = pt[c1] / pt[c2]
    
    
    
    if drop_oid:
        pt.reset_index(drop=True, inplace=True)
    else:
        pt.reset_index(inplace=True)
    pt.add_prefix(PREF+'_').to_pickle(output_path)
    
    return

def multi(args):
    input_path, output_path = args
    aggregate(pd.read_pickle(input_path), output_path, drop_oid=False)
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    aggregate(pd.read_pickle('../data/train_log.pkl'), f'../data/train_{PREF}.pkl')
    
    # test
    os.system(f'rm ../data/tmp_{PREF}*')
    argss = []
    for i,file in enumerate(utils.TEST_LOGS):
        argss.append([file, f'../data/tmp_{PREF}{i}.pkl'])
    pool = Pool( cpu_count() )
    pool.map(multi, argss)
    pool.close()
    df = pd.concat([pd.read_pickle(f) for f in glob(f'../data/tmp_{PREF}*')], 
                    ignore_index=True)
    df.sort_values(f'{PREF}_object_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    del df[f'{PREF}_object_id']
    df.to_pickle(f'../data/test_{PREF}.pkl')
    os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)


