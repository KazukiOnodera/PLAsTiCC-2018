#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:41:16 2018

@author: kazuki.onodera


                      pb0       pb1       pb2       pb3       pb4       pb5
object_id date                                                             
615       59750       NaN -1.235849 -0.890235 -1.057542 -1.018409       NaN
          59752       NaN -1.606743 -1.114177 -1.177723 -1.030178 -0.941031
          59767       NaN -1.233963 -0.895470 -1.066808 -1.062076 -1.113729
          59770       NaN -1.241311 -0.906727 -1.070148 -1.047955 -1.098096
          59779       NaN -1.394135 -1.030293 -1.163318 -1.105329 -1.118002
          59782       NaN -0.679803 -0.457592 -0.710519 -0.871533 -0.965328
          59797       NaN  0.053755  0.639557  0.741746  0.943564  0.976867
          59800       NaN  0.196090  0.275726  0.067575 -0.159554 -0.340889
          59807       NaN -0.636966 -0.419391 -0.670657 -0.816795 -0.911018
          59810       NaN -0.797759 -0.560177 -0.815015 -0.912751 -1.034594
          59813       NaN -1.665753 -1.107946 -1.136741 -0.796038 -0.495219


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

stats = ['min', 'max', 'mean', 'median', 'std', quantile(25), quantile(75)]


keys = ['object_id']

def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_feather('../data/train_log.f')
    """
    
    df['date'] = df.mjd.astype(int)
    
#    df.flux += df.groupby(keys).flux.transform('min').abs()
    df.flux /= df.groupby(keys).flux.transform('max')
    
    pt = pd.pivot_table(df, index=['object_id', 'date'], columns=['passband'], values=['flux'])
    pt.columns = pd.Index([f'pb{e[1]}' for e in pt.columns.tolist()])
    
    col = pt.columns.tolist()
    comb = list(combinations(col, 2))
    li = []; num_aggregations = {}
    for c1,c2 in comb:
        # diff
        c = f'{c1}-m-{c2}' 
        pt[c] = pt[c1] - pt[c2]
        li.append(c)
        num_aggregations[c] = stats
        
        # ratio
        c = f'{c1}-d-{c2}' 
        pt[c] = pt[c1] / pt[c2]
        li.append(c)
        num_aggregations[c] = stats
    
    pt['pb_min'] = pt[col].min(1)
    num_aggregations['pb_min'] = stats
    pt['pb_max'] = pt[col].max(1)
    num_aggregations['pb_max'] = stats
    pt['pb_sum'] = pt[col].sum(1)
    num_aggregations['pb_sum'] = stats
    pt['pb_mean'] = pt[col].mean(1)
    num_aggregations['pb_mean'] = stats
    pt['pb_std'] = pt[col].std(1)
    num_aggregations['pb_std'] = stats
    pt['pb_median'] = pt[col].median(1)
    num_aggregations['pb_median'] = stats
    
    pt.reset_index(inplace=True)
    
    # =============================================================================
    # diff agg
    # =============================================================================
    pt_diff = pt.diff().add_suffix('_diff')
    pt_diff.loc[pt['object_id'] != pt['object_id'].shift()] = np.nan
    pt_diff.drop(['object_id_diff', 'date_diff'], axis=1, inplace=True)
    pt_diff['object_id'] = pt['object_id']
    num_aggregations2 = {}
    for c in pt_diff.columns[1:]:
        num_aggregations2[c] = stats
    
    df_agg = pt_diff.groupby('object_id').agg(num_aggregations2)
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    # std / mean
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    # max / min
    col_max = [c for c in df_agg.columns if c.endswith('_max')]
    for c in col_max:
        df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
    
    df_agg.reset_index(drop=True, inplace=True)
    diff_agg = df_agg
    
    # =============================================================================
    # agg
    # =============================================================================
    df_agg = pt.groupby(['object_id']).agg(num_aggregations)
    df_agg.columns = pd.Index([f'{e[0]}_{e[1]}' for e in df_agg.columns.tolist()])
    
    # std / mean
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    # max / min
    col_max = [c for c in df_agg.columns if c.endswith('_max')]
    for c in col_max:
        df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
    
    if drop_oid:
        df_agg.reset_index(drop=True, inplace=True)
    else:
        df_agg.reset_index(inplace=True)
    df_agg = pd.concat([df_agg, diff_agg], axis=1)
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


