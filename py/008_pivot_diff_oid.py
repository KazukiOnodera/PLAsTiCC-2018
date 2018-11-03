#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:59:09 2018

@author: kazuki.onodera

diff

keys: object_id, passband


"""


import numpy as np
import pandas as pd
import os
from glob import glob
from multiprocessing import cpu_count, Pool
import utils

PREF = 'f008'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

def quantile(n):
    def quantile_(x):
        return np.percentile(x, n)
    quantile_.__name__ = 'q%s' % n
    return quantile_

stats = ['min', 'max', 'mean', 'median', 'sum', 'std', quantile(25), quantile(75)]

usecols = ['flux', 'flux_norm1', 'flux_norm2', 'flux_err', 'detected']

def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    pt = pd.pivot_table(df, index=['object_id', 'date'], columns=['passband'], 
                        values=usecols)
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
        
        num_aggregations = {f'{c}_diff'    : stats,
                            f'{c}_diff_abs': stats,
                            f'{c}_chng'    : stats,
                            f'{c}_chng_abs': stats,
                            }
        
        df_agg = tmp.groupby('object_id').agg(num_aggregations)
        df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
        
        # std / mean
        col_std = [c for c in df_agg.columns if c.endswith('_std')]
        for c in col_std:
            df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
        
        # max / min, max - min
        col_max = [c for c in df_agg.columns if c.endswith('_max')]
        for c in col_max:
            df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
            df_agg[f'{c}-m-min'] = df_agg[c]-df_agg[c.replace('_max', '_min')]
            
        feature.append(df_agg)
    
    df_agg = pd.concat(feature, axis=1)
    
    if drop_oid:
        df_agg.reset_index(drop=True, inplace=True)
    else:
        df_agg.reset_index(inplace=True)
    df_agg.add_prefix(PREF+'_').to_pickle(output_path)
    
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
    utils.stop_instance()

