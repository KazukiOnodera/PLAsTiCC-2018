#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 02:01:23 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
from glob import glob
from scipy.stats import kurtosis
from multiprocessing import cpu_count, Pool

import sys
argvs = sys.argv

from itertools import combinations
import utils

PREF = 'f027'


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


num_aggregations = {
    'flux':        stats,
    'flux_norm1':  stats,
    'flux_norm2':  stats,
    'flux_norm3':  stats,
    'flux_err':    stats,
    'detected':    stats,
    'flux_ratio_sq': stats,
    'flux_by_flux_ratio_sq': stats,
    'lumi': stats,
    }

def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    df = df[df.flux_norm3==1]
    
    df.mjd = df.mjd.diff()
    df.loc[df.object_id != df.object_id.shift(), 'mjd'] = np.nan
    
    df['order'] = 1
    df['order'] = df.groupby('object_id')['order'].cumsum()
    
    pt1 = pd.pivot_table(df, index=['object_id'], columns=['order'],
                        values=['mjd', 'passband'])
    
    pt1.columns = pd.Index([f'order{e[1]}_{e[0]}' for e in pt1.columns.tolist()])
    
    
    pt2 = pd.pivot_table(df, index=['object_id'], 
                         aggfunc=num_aggregations)
    pt2.columns = pd.Index([f'{e[0]}_{e[1]}' for e in pt2.columns.tolist()])
    
    
    pt = pd.concat([pt1, pt2], axis=1)
    
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
    if utils.GENERATE_TEST:
        imp = pd.read_csv(utils.IMP_FILE).head(utils.GENERATE_FEATURE_SIZE)
        usecols = imp[imp.feature.str.startswith(f'{PREF}')][imp.gain>0].feature.tolist()
#        usecols = [c.replace(f'{PREF}_', '') for c in usecols]
#        usecols += ['object_id']
        
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
        utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
        utils.save_test_features(df[usecols])
        os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)

