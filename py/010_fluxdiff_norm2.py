#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:41:16 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
from itertools import combinations
import utils

PREF = 'f010'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')


stats = ['min', 'max', 'mean', 'median', 'std']


keys = ['object_id', 'passband']

def aggregate(df, output_path):
    """
    df = pd.read_feather('../data/train_log.f')
    """
    
    df['date'] = df.mjd.astype(int)
    
    df.flux += df.groupby(keys).flux.transform('min').abs()
    df.flux /= df.groupby(keys).flux.transform('max')
    
    pt = pd.pivot_table(df, index=['object_id', 'date'], columns=['passband'], values=['flux'])
    pt.columns = pd.Index([f'pb{e[1]}' for e in pt.columns.tolist()])
    
    col = pt.columns.tolist()
    comb = list(combinations(col, 2))
    li = []; num_aggregations = {}
    for c1,c2 in comb:
        c = f'{c1}-m-{c2}' 
        pt[c] = pt[c1] - pt[c2]
        li.append(c)
        num_aggregations[c] = stats
    
    pt.reset_index(inplace=True)
    
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

