#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:12:10 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os, gc
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

import utils

os.system(f'rm -rf ../data')
os.system(f'mkdir ../data')
os.system(f'rm -rf ../feature')
os.system(f'mkdir ../feature')

COLUMN_TO_TYPE = {
    'object_id': np.int32,
    'mjd'      : np.float32,
    'passband' : np.int8,
    'flux'     : np.float32,
    'flux_err' : np.float32,
    'detected' : np.int8
}

def preprocess(df):
    df['date'] = df.mjd.astype(int)
    
    df['year'] = ( df.date - df.groupby(['object_id']).date.transform('min') )/365
    df['year'] = df['year'].astype(int)
    
    df['month'] = ( df.date - df.groupby(['object_id']).date.transform('min') )/30
    df['month'] = df['month'].astype(int)
    
    df['3month'] = ( df.date - df.groupby(['object_id']).date.transform('min') )/90
    df['3month'] = df['3month'].astype(int)
    
    df['flux_norm1'] = df.flux / df.groupby(['object_id']).flux.transform('max')
    df['flux_norm2'] = (df.flux - df.groupby(['object_id']).flux.transform('min')) / df.groupby(['object_id']).flux.transform('max')
    
    return

def multi(splitn):
    df = test_log[test_log.object_id%utils.SPLIT_SIZE == splitn].reset_index(drop=True)
    preprocess(df)
    df.to_pickle(f'../data/test_log{splitn:02}.pkl')
    return

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
    preprocess(train_log)
    train_log.to_pickle('../data/train_log.pkl')
    
    
    # =================
    # test
    # =================
    test     = pd.read_csv('../input/test_set_metadata.csv.zip')
    test[test['hostgal_photoz'] == 0][['object_id']].reset_index(drop=True).to_pickle('../data/oid_gal.pkl')
    test[test['hostgal_photoz'] != 0][['object_id']].reset_index(drop=True).to_pickle('../data/oid_exgal.pkl')
    
    
    test.to_pickle('../data/test.pkl')
    
    test_log = pd.read_csv('../input/test_set.csv.zip', dtype=COLUMN_TO_TYPE)
    
#    pool = Pool(cpu_count())
#    pool.map(multi, range(utils.SPLIT_SIZE))
#    pool.close()
    
    for i in tqdm(range(utils.SPLIT_SIZE), mininterval=15):
        gc.collect()
        df = test_log[test_log.object_id%utils.SPLIT_SIZE==i].reset_index(drop=True)
        preprocess(df)
        df.to_pickle(f'../data/test_log{i:02}.pkl')
    
    utils.end(__file__)

