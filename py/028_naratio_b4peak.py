#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:43:36 2018

@author: Kazuki

最大光度前の欠損率

"""


import numpy as np
import pandas as pd
import os
from glob import glob
#from scipy.stats import kurtosis
from multiprocessing import cpu_count, Pool

import sys
argvs = sys.argv

from itertools import combinations
import utils

PREF = 'f028'


os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')



def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(9999)
    """
    
    gr = df.groupby('object_id')
    
    feature = gr.date.min()
    feature.name = 'date_min'
    feature = feature.to_frame()
    feature['date_flux_max'] = df.iloc[gr.flux.idxmax()].set_index('object_id')['date']
    feature['date_diff_highest-min'] = feature['date_flux_max'] - feature['date_min']
    
    dates = [30, 50, 70, 100, 200, 300]
    for d in dates:
        # -d ~highest date
        idxmax = gr.flux.idxmax()
        base = df.iloc[idxmax][['object_id', 'date']]
        li = []
        for i in range(d):
            i += 1
            lag = base.copy()
            lag['date'] -= i
            li.append(lag)
        
        keep = pd.concat(li)
        
        df_ = pd.merge(keep, df, on=['object_id', 'date'], how='inner')
        feature[f'{d}_notna_ratio_b4peak'] = df_.groupby(['object_id', 'date']).size().groupby(['object_id']).size() / d # faster than 'date.nunique()'
        feature[f'{d}_date_max'] = df_.groupby('object_id').date.max()
        feature[f'{d}_date_diff'] = feature['date_flux_max'] - feature[f'{d}_date_max']
    
    
    
    # ======== okumura features ========
    det_mjd_diff = df[df['detected']==1].pivot_table('mjd','object_id',aggfunc=[min,max])
    det_mjd_diff.columns = ['min_mjd','max_mjd']
    
    # detected==1の前後の間隔を追加
    mjd_diff_ = df[['object_id','mjd']].merge(right=det_mjd_diff, on=['object_id'], how='left')
    
    max_mjd_bf_det1 = mjd_diff_[mjd_diff_.mjd < mjd_diff_.min_mjd].groupby('object_id')[['mjd', 'min_mjd']].max().rename(columns={'mjd': 'max_mjd_bf_det1'})
    feature['mjd_diff_bf_det1'] = max_mjd_bf_det1['min_mjd'] - max_mjd_bf_det1['max_mjd_bf_det1']
    
    min_mjd_af_det1 = mjd_diff_[mjd_diff_.mjd > mjd_diff_.max_mjd].groupby('object_id')[['mjd', 'max_mjd']].min().rename(columns={'mjd': 'min_mjd_af_det1'})
    feature['mjd_diff_af_det1'] = min_mjd_af_det1['min_mjd_af_det1'] - min_mjd_af_det1['max_mjd'] 
    
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

