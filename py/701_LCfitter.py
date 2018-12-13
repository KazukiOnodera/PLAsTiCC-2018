#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 05:27:59 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils

PREF = 'f701'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    # train
    tr = utils.load_train(['object_id'])
    df = pd.read_pickle('../FROM_MYTEAM/LCfit_feature_allSN_i_train_v1_20181212.pkl.gz')
    df = pd.merge(tr, df, on='object_id', how='left')
    df.reset_index(drop=True, inplace=True)
    del df['object_id']
    df.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')
    
    # test
    te = utils.load_test(['object_id'])
    df = pd.read_pickle('../FROM_MYTEAM/LCfit_feature_allSN_i_test_v1_20181212.pkl.gz')
    df = pd.merge(te, df, on='object_id', how='left')
    df.reset_index(drop=True, inplace=True)
    del df['object_id']
    df = df.add_prefix(PREF+'_')
    utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
    
    utils.end(__file__)

