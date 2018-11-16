#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 02:36:34 2018

@author: Kazuki

oid, pb, month(30days) で統計量を取って変化率を出す


"""

import numpy as np
import pandas as pd
import os
from glob import glob
from scipy.stats import kurtosis
from multiprocessing import cpu_count, Pool
from tsfresh.feature_extraction import extract_features

import sys
argvs = sys.argv

from itertools import combinations
import utils

PREF = 'f019'

if len(argvs)>1:
    is_test = int(argvs[1])
else:
    is_test = 0


os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

def quantile(n):
    def quantile_(x):
        return np.percentile(x, n)
    quantile_.__name__ = 'q%s' % n
    return quantile_

def kurt(x):
    return kurtosis(x)

stats = ['min', 'max', 'mean', 'median', 'std','skew',
         kurt, quantile(25), quantile(75)]


num_aggregations = {
    'flux':        stats,
    'flux_norm1':  stats,
    'flux_norm2':  stats,
    'flux_err':    stats,
    'detected':    stats,
    'flux_ratio_sq': stats,
    'flux_by_flux_ratio_sq': stats,
    }

fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},
                           {'coeff': 1, 'attr': 'abs'}],
        'kurtosis' : None, 'skewness' : None}

def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    # 
    idxmax = df.groupby('object_id').flux.idxmax()
    base = df.iloc[idxmax][['object_id', 'date']]
    li = [base]
    for i in range(30*6):
        i += 1
        lead = base.copy()
        lead['date'] += i
        li.append(lead)
    
    keep = pd.concat(li)
    keep.sort_values(['object_id', 'date'], inplace=True)
    keep['month'] = (keep.date - keep.groupby('object_id').date.transform('min'))/30
    keep['month'] = keep['month'].astype(int)
    
    del df['month'] # already exist
    
    df = pd.merge(keep, df, on=['object_id', 'date'], how='inner')
    
    pt = pd.pivot_table(df, index=['object_id'], columns=['month', 'passband'],
                        aggfunc=num_aggregations)
    
    pt.columns = pd.Index([f'm{e[2]}_pb{e[3]}_{e[0]}_{e[1]}' for e in pt.columns.tolist()])
    
    # compare month
    col = pd.Series([f'{c[2:]}' for c in pt.columns if c.startswith('m0_')])
    for c1,c2 in list(combinations(range(6), 2)):
        col1 = (f'm{c1}'+col).tolist()
        col2 = (f'm{c2}'+col).tolist()
        for c1,c2 in zip(col1, col2):
            pt[f'{c1}-d-{c2}'] = pt[c1] / pt[c2]
    
    
    if usecols is not None:
        col = [c for c in pt.columns if c not in usecols]
        pt.drop(col, axis=1, inplace=True)
    
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
    
    usecols = None
    aggregate(pd.read_pickle('../data/train_log.pkl'), f'../data/train_{PREF}.pkl')
    
    # test
    if is_test:
        imp = pd.read_csv(utils.IMP_FILE).head(utils.GENERATE_FEATURE_SIZE)
        usecols = imp[imp.feature.str.startswith(f'{PREF}')][imp.gain>0].feature.tolist()
        usecols = [c.replace(f'{PREF}_', '') for c in usecols]
        usecols += ['object_id']
        
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
