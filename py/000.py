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
    
    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    
    df['date'] = df.mjd.astype(int)
    
    df['year'] = ( df.date - df.groupby(['object_id']).date.transform('min') )/365
    df['year'] = df['year'].astype(int)
    
#    df['month'] = ( df.date - df.groupby(['object_id']).date.transform('min') )/30
#    df['month'] = df['month'].astype(int)
#    
#    df['3month'] = ( df.date - df.groupby(['object_id']).date.transform('min') )/90
#    df['3month'] = df['3month'].astype(int)
    
    df['flux_norm1'] = df.flux / df.groupby(['object_id']).flux.transform('max')
    df['flux_norm2'] = (df.flux - df.groupby(['object_id']).flux.transform('min')) / df.groupby(['object_id']).flux.transform('max')
    df['flux_norm3'] = df.flux / df.groupby(['object_id', 'passband']).flux.transform('max')
    
    return

def multi(splitn):
    df = test_log[test_log.object_id%utils.SPLIT_SIZE == splitn].reset_index(drop=True)
    preprocess(df)
    df.to_pickle(f'../data/test_log{splitn:02}.pkl')
    return

def ddf_to_wfd(df, n, oid_start):
    """
    te.object_id.max()
    130788054
    """
    df['object_id_bk'] = df.object_id.copy()
    df['object_id'] = df.object_id.rank(method='dense')
#    oid_start = oid_max + 1
    li = []
    for i in tqdm(range(n)):
        tmp = df.sample(frac=1, random_state=i).drop_duplicates(['object_id', 'date'])
        tmp.object_id += oid_start
        oid_start = tmp.object_id.max() #+ 1
#        print(tmp.object_id.min(), tmp.object_id.max())
        li.append(tmp)
    
    df = pd.concat(li, ignore_index=True)
    meta = df[train.columns.tolist()+['object_id_bk']].drop_duplicates('object_id')
    log = df[train_log.columns]
    
    meta.ddf = 0
    
    return meta, log, meta.object_id.max()

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    # =================
    # train
    # =================
    train     = pd.read_csv('../input/training_set_metadata.csv')
    (train['hostgal_photoz'] == 0).to_pickle('../data/tr_is_gal.pkl')
    train[train['hostgal_photoz'] == 0][['object_id']].reset_index(drop=True).to_pickle('../data/tr_oid_gal.pkl')
    train[train['hostgal_photoz'] != 0][['object_id']].reset_index(drop=True).to_pickle('../data/tr_oid_exgal.pkl')
    train.to_pickle('../data/train.pkl')
    train[['target']].to_pickle('../data/target.pkl')
    
    train_log = pd.read_csv('../input/training_set.csv.zip', dtype=COLUMN_TO_TYPE)
    train_log = pd.merge(train_log, train[['object_id', 'distmod']], on='object_id', how='left')
    train_log['lumi'] = train_log['flux'] * 4 * np.pi * 10 ** ((train_log['distmod']+5)/2.5)
    del train_log['distmod']
    preprocess(train_log)
    train_log.to_pickle('../data/train_log.pkl')
    
    # =================
    # data augment
    # =================
    if utils.GENERATE_AUG:
        train_log_ddf = pd.merge(train_log, train[train.ddf==1], how='inner', on='object_id')
        train_log_ddf_gal   = train_log_ddf[train_log_ddf.hostgal_photoz==0] # 520
        train_log_ddf_exgal = train_log_ddf[train_log_ddf.hostgal_photoz!=0] # 37
        
        meta_wfd_gal,   log_wfd_gal,   oid_max = ddf_to_wfd(train_log_ddf_gal, 520,  130788054)
        meta_wfd_exgal, log_wfd_exgal, oid_max = ddf_to_wfd(train_log_ddf_exgal, 37, oid_max)
        
        meta_wfd = pd.concat([meta_wfd_gal, meta_wfd_exgal], ignore_index=True)
        meta_wfd.sort_values('object_id', inplace=True)
        meta_wfd.reset_index(drop=True, inplace=True)
        meta_wfd.to_pickle('../data/train_aug.pkl')
        meta_wfd[['target']].to_pickle('../data/target_aug.pkl')
        
        log_wfd = pd.concat([log_wfd_gal, log_wfd_exgal], ignore_index=True)
        for i in tqdm(range(utils.SPLIT_SIZE), mininterval=15):
            gc.collect()
            df = log_wfd[log_wfd.object_id%utils.SPLIT_SIZE==i].reset_index(drop=True)
            df.to_pickle(f'../data/train_log_aug{i:02}.pkl')
    
    # =================
    # test
    # =================
    test     = pd.read_csv('../input/test_set_metadata.csv.zip')
    (test['hostgal_photoz'] == 0).to_pickle('../data/te_is_gal.pkl')
    test[test['hostgal_photoz'] == 0][['object_id']].reset_index(drop=True).to_pickle('../data/te_oid_gal.pkl')
    test[test['hostgal_photoz'] != 0][['object_id']].reset_index(drop=True).to_pickle('../data/te_oid_exgal.pkl')
    
    
    test.to_pickle('../data/test.pkl')
    
    test_log = pd.read_csv('../input/test_set.csv.zip', dtype=COLUMN_TO_TYPE)
    
    test_log = pd.merge(test_log, test[['object_id', 'distmod']], on='object_id', how='left')
    test_log['lumi'] = test_log['flux'] * 4 * np.pi * 10 ** ((test_log['distmod']+5)/2.5)
    del test_log['distmod']
    
    for i in tqdm(range(utils.SPLIT_SIZE), mininterval=15):
        gc.collect()
        df = test_log[test_log.object_id%utils.SPLIT_SIZE==i].reset_index(drop=True)
        preprocess(df)
        df.to_pickle(f'../data/test_log{i:02}.pkl')
    
    utils.end(__file__)

