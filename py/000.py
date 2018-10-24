#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:12:10 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

import utils

os.system(f'rm -rf ../data')
os.system(f'mkdir ../data')
os.system(f'rm -rf ../feature')
os.system(f'mkdir ../feature')

COLUMN_TO_TYPE = {
    'object_id': np.int32,
    'mjd': np.float32,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.int8
}

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    # =================
    # train
    # =================
    train     = pd.read_csv('../input/training_set_metadata.csv')
    train.to_pickle('../data/train.pkl')
    train[['target']].to_pickle('../data/target.pkl')
    
    train_log = pd.read_csv('../input/training_set.csv.zip', dtype=COLUMN_TO_TYPE)
    train_log.to_pickle('../data/train_log.pkl')
    
    
    # =================
    # test
    # =================
    test     = pd.read_csv('../input/test_set_metadata.csv.zip')
    test.to_pickle('../data/test.pkl')
    
    test_log = pd.read_csv('../input/test_set.csv.zip', dtype=COLUMN_TO_TYPE)
    
    for i in tqdm(range(utils.SPLIT_SIZE), mininterval=15):
        test_log[test_log.object_id%utils.SPLIT_SIZE==i].reset_index(drop=True).to_pickle(f'../data/test_log{i:02}.pkl')
    
    utils.end(__file__)

