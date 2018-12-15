#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 06:10:42 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils

PREF = 'f711'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

sn_classes = [42, 52, 62, 67, 90]
suffix_list = ['fmax', 'redchisq', 'chisq', 'dmax', 'dof']


def get_feature(df):
    col_init = df.columns
    for suf in suffix_list:
        col = [c for c in col_init if c.endswith(suf)]
        df[f'{suf}_min'] = df[col].min(1)
        df[f'{suf}_mean'] = df[col].mean(1)
        df[f'{suf}_max'] = df[col].max(1)
        df[f'{suf}_std'] = df[col].std(1)
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    # train
    tr = utils.load_train(['object_id'])
    
    df = pd.read_pickle('../FROM_MYTEAM/LCfit_feature_allSN_i_train_v3_20181215.pkl.gz')
    df = pd.merge(tr, df, on='object_id', how='left')
    df.reset_index(drop=True, inplace=True)
    get_feature(df)
    
    del df['object_id']
    df.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')
    
    # test
    te = utils.load_test(['object_id'])
    df = pd.read_pickle('../FROM_MYTEAM/LCfit_feature_allSN_i_test_v3_20181215.pkl.gz')
    df = pd.merge(te, df, on='object_id', how='left')
    df.reset_index(drop=True, inplace=True)
    get_feature(df)
    
    del df['object_id']
    df = df.add_prefix(PREF+'_')
    utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
    
    utils.end(__file__)

