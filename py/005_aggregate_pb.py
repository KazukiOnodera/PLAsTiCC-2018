#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:55:39 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
#from multiprocessing import Pool
import utils

PREF = 'f005'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

num_aggregations = {
    'mjd':      ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std'],
    'flux':     ['min', 'max', 'mean', 'median', 'std'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std'],
    'detected': ['min', 'max', 'mean', 'median', 'std'],
    }

def aggregate(df, output_path):
    
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
    
    
    num_aggregations2 = {}
    for c in df_agg.columns:
        num_aggregations2[c] = ['min', 'max', 'mean', 'median', 'std']
    
    df_agg.reset_index(inplace=True)
    
    df_agg = df_agg.groupby(['object_id']).agg(num_aggregations2)
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg.reset_index(drop=True, inplace=True)
    df_agg.add_prefix(PREF+'_').to_feather(output_path)
    
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    aggregate(pd.read_feather('../data/train_log.f'), f'../data/train_{PREF}.f')
    aggregate(pd.read_feather('../data/test_log.f'),  f'../data/test_{PREF}.f')
    
    utils.end(__file__)

