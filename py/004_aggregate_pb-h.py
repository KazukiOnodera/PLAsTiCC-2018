#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:29:28 2018

@author: Kazuki

horizon

"""

import numpy as np
import pandas as pd
import os
from glob import glob
from multiprocessing import cpu_count, Pool
import utils

PREF = 'f004'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

def quantile(n):
    def quantile_(x):
        return np.percentile(x, n)
    quantile_.__name__ = 'q%s' % n
    return quantile_

num_aggregations = {
    'mjd':      ['min', 'max', 'size'],
    'flux':     ['min', 'max', 'mean', 'median', 'std', quantile(25), quantile(75)],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', quantile(25), quantile(75)],
    'detected': ['min', 'max', 'mean', 'median', 'std', quantile(25), quantile(75)],
    }

def aggregate(df, output_path):
    
#    df, output_path = args
    df_agg = df.groupby(['object_id', 'passband']).agg(num_aggregations)
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    # std / mean
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    # max / min
    col_max = [c for c in df_agg.columns if c.endswith('_max')]
    for c in col_max:
        df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
    
    df_agg.reset_index(inplace=True)
    df_ = pd.pivot_table(df_agg, index=['object_id'], columns=['passband'])
    df_.columns = pd.Index([f'pb{e[1]}_{e[0]}' for e in df_.columns.tolist()])
    
    df_.reset_index(drop=True, inplace=True)
    df_.add_prefix(PREF+'_').to_feather(output_path)
    
    return

def multi(args):
    input_path, output_path = args
    aggregate(pd.read_feather(input_path), output_path, drop_oid=False)
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    aggregate(pd.read_feather('../data/train_log.f'), f'../data/train_{PREF}.f')
    
    # test
    os.system(f'rm ../data/tmp*')
    argss = []
    for i,file in enumerate(utils.TEST_LOGS):
        argss.append([file, f'../data/tmp{i}.f'])
    pool = Pool( cpu_count() )
    pool.map(multi, argss)
    pool.close()
    df = pd.concat([pd.read_feather(f) for f in glob('../data/tmp*')], 
                    ignore_index=True)
    df.sort_values(f'{PREF}_object_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    del df[f'{PREF}_object_id']
    df.to_feather(f'../data/test_{PREF}.f')
    os.system(f'rm ../data/tmp*')
    
    
    utils.end(__file__)

