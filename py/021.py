#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 04:08:53 2018

@author: Kazuki

ピークからの変化率

keys: object_id, passband


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

import utils

PREF = 'f021'

if len(argvs)>1:
    is_test = int(argvs[1])
else:
    is_test = 0

max_index = 30

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
         kurt, quantile(10), quantile(25), quantile(75), quantile(90)]

num_aggregations1 = {
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
    tr = utils.load_train()
    oids = tr[tr.target==15].object_id
    
    df = pd.read_pickle('../data/train_log.pkl').head(9999)
    
    df = df[df.object_id.isin(oids)].reset_index(drop=True)
    
    """
    
    # peak date ~ 60
    idxmax = df.groupby('object_id').flux.idxmax()
    base = df.iloc[idxmax][['object_id', 'date']]
    li = [base]
    for i in range(60):
        i += 1
        lead = base.copy()
        lead['date'] += i
        li.append(lead)
    
    keep = pd.concat(li)
    
    df = pd.merge(keep, df, on=['object_id', 'date'], how='inner')
    
    pt = pd.pivot_table(df, index=['object_id', 'date'], columns=['passband'], 
                        values=list(num_aggregations1.keys()))
#                        aggfunc=num_aggregations)
    
    pt.columns = pd.Index([f'pb{e[1]}_{e[0]}' for e in pt.columns.tolist()])
    pt.reset_index(inplace=True)
    
    feature = []
    col = pt.columns
    for c in col[2:]:
        
        df_diff = pt[['object_id', 'date', c]].dropna().diff().add_suffix('_diff')
        df_diff_abs = df_diff.abs().add_suffix('_abs')
        
        df_chng = pt[['object_id', 'date', c]].dropna().pct_change().add_suffix('_chng')
        df_chng_abs = df_chng.abs().add_suffix('_abs')
        
        
        tmp = pd.concat([df_diff, df_diff_abs, df_chng, df_chng_abs], axis=1)
        tmp.loc[pt['object_id'] != pt['object_id'].shift()] = np.nan
        col2 = [c for c in tmp.columns if 'object_id' in c]
        tmp.drop(col2, axis=1, inplace=True)
        tmp['object_id'] = pt['object_id']
        
        tmp[f'{c}_diff']     /= tmp['date_diff'] # change rate per day
        tmp[f'{c}_diff_abs'] /= tmp['date_diff']
        tmp[f'{c}_chng']     /= tmp['date_diff']
        tmp[f'{c}_chng_abs'] /= tmp['date_diff']
        
        tmp['ind'] = 1
        tmp['ind'] = tmp.groupby('object_id')['ind'].cumsum()
        tmp = tmp[tmp['ind']<=max_index]
        
        col_drop = [c for c in tmp.columns if 'date' in c]
        if c.endswith('detected'):
            pb = c[:3]
            tmp[f'pb{pb}_date_diff'] = tmp['date_diff']
        tmp.drop(col_drop, axis=1, inplace=True)
        
        tmp = pd.pivot_table(tmp, index=['object_id'], columns=['ind'], )
        
        feature.append(tmp)
    
    pt = pd.concat(feature, axis=1)
    pt.columns = pd.Index([f'{e[0]}_date{e[1]}' for e in pt.columns.tolist()])
    
    
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
        utils.save_test_features(df)
        os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)

