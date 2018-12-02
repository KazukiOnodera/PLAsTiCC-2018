#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 06:35:29 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
from glob import glob
from multiprocessing import cpu_count, Pool

import sys
argvs = sys.argv

from itertools import combinations
import utils

PREF = 'f026'


os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')


def date_diff(df, name):
    feature = df.groupby('object_id').size().to_frame()
    del feature[0]
    feature[f'date_diff_{name}'] = df.groupby('object_id').date.max() - df.groupby('object_id').date.min()
    
    tmp = df.groupby(['object_id', 'passband']).date.max() - df.groupby(['object_id', 'passband']).date.min()
    tmp.name = f'date_diff_{name}'
    tmp = pd.pivot_table(tmp.reset_index(), index=['object_id'], columns=['passband'], 
                         values=[f'date_diff_{name}'])
    tmp.columns = pd.Index([f'pb{e[1]}_{e[0]}' for e in tmp.columns.tolist()])
    
    return pd.concat([feature, tmp], axis=1)

def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    feature = df.groupby('object_id').size().to_frame()
    del feature[0]
    
#    # highest -30 ~ date ~ 30
#    idxmax = df.groupby('object_id').flux.idxmax()
#    base = df.iloc[idxmax][['object_id', 'date']]
#    li = [base]
#    for i in range(1, 31):
#        lag  = base.copy()
#        lead = base.copy()
#        lag['date']  -= i
#        lead['date'] += i
#        li.append(lag)
#        li.append(lead)
#    
#    keep = pd.concat(li)
#    
#    df = pd.merge(keep, df, on=['object_id', 'date'], how='inner')
    
    df_ = df[df.flux_norm1 > 0.3]
    feature1 = date_diff(df_, 0.3)
    
    df_ = df[df.flux_norm1 > 0.5]
    feature2 = date_diff(df_, 0.5)
    
    df_ = df[df.flux_norm1 > 0.7]
    feature3 = date_diff(df_, 0.7)
    
    df_ = df[df.flux_norm1 > 0.9]
    feature4 = date_diff(df_, 0.9)
    
    feature = pd.concat([feature1, feature2, feature3, feature4], axis=1)
    
    if drop_oid:
        feature.reset_index(drop=True, inplace=True)
    else:
        feature.reset_index(inplace=True)
    feature.add_prefix(PREF+'_').to_pickle(output_path)
    
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
    
    if utils.GENERATE_AUG:
        os.system(f'rm ../data/tmp_{PREF}*')
        argss = []
        for i,file in enumerate(utils.AUG_LOGS):
            argss.append([file, f'../data/tmp_{PREF}{i}.pkl'])
        pool = Pool( cpu_count() )
        pool.map(multi, argss)
        pool.close()
        df = pd.concat([pd.read_pickle(f) for f in glob(f'../data/tmp_{PREF}*')], 
                        ignore_index=True)
        df.sort_values(f'{PREF}_object_id', inplace=True)
        df.reset_index(drop=True, inplace=True)
        del df[f'{PREF}_object_id']
        utils.to_pkl_gzip(df, f'../data/train_aug_{PREF}.pkl')
        os.system(f'rm ../data/tmp_{PREF}*')
    
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




