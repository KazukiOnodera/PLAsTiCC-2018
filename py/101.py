#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:43:32 2018

@author: Kazuki





In [17]: tr_log
Out[17]: 
         object_id   date  passband         flux  flux_norm1  flux_norm2
0              615  59748         1  -816.434326   -1.235849    0.429904
1              615  59748         2  -544.810303   -0.824688    0.841065
2              615  59748         3  -471.385529   -0.713543    0.952209
"""

import numpy as np
import pandas as pd
import os
import utils

def augment(df, n):
    if n <= 0:
        raise
    li = []
    for i in range(1, n+1):
        tmp = df.copy()
        tmp['mjd'] += i
        li.append(tmp)
        tmp = df.copy()
        tmp['mjd'] -= i
        li.append(tmp)
    tmp = pd.concat(li)
    df = pd.concat([df, tmp], ignore_index=True)
    df.date = df.mjd.astype(int)
    return df


def preprocess(df):
    
    df['flux_norm1'] = df.flux / df.groupby(['object_id']).flux.transform('max')
    df['flux_norm2'] = (df.flux - df.groupby(['object_id']).flux.transform('min')) / df.groupby(['object_id']).flux.transform('max')
    

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    # =============================================================================
    # train
    # =============================================================================
    tr_log = pd.read_pickle('../data/train_log.pkl')
    
    tr_log = augment(tr_log, 2)
    
    tr_log = tr_log.groupby(['object_id', 'date', 'passband']).flux.mean().reset_index()
    
    preprocess(tr_log)
    
    tr_log.to_pickle('../data/train_log2.pkl')
    
    # test
    if utils.GENERATE_TEST:
        pass
    
    utils.end(__file__)

