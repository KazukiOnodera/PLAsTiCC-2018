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
    df = pd.read_pickle('../FROM_MYTEAM/LCfit_features_train_20181129.pkl.gz')
    df.sort_values('object_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    del df['object_id']
    df.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')
    
    # test
    df = pd.read_pickle('../FROM_MYTEAM/LCfit_features_test_20181130.pkl.gz')
    df.sort_values('object_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    del df['object_id']
    df = df.add_prefix(PREF+'_')
    utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
    
    utils.end(__file__)

