#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:29:46 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
from glob import glob
from scipy.stats import kurtosis
from multiprocessing import cpu_count, Pool
#from tsfresh.feature_extraction import extract_features

import sys
argvs = sys.argv

import utils

PREF = 'ftmp2'


os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

DAYS = 10


# =============================================================================
# def
# =============================================================================

class_SN = [42, 52, 62, 87, 90]

tr = pd.read_pickle('../data/train.pkl')


tr_log = pd.read_pickle('../data/train_log.pkl')
tr_log = pd.merge(tr_log, tr[['object_id', 'hostgal_photoz', 'target']], 
                  on='object_id', how='left')
tr_log = tr_log[tr_log.target.isin(class_SN)].reset_index(drop=True)

# after peak for 60days
idxmax = tr_log.groupby('object_id').flux.idxmax()
tbl = tr_log.iloc[idxmax].reset_index(drop=True)
tbl['date_start'] = tbl['date']
tbl['date_end'] = tbl['date'] + DAYS
tr_log = pd.merge(tr_log, tbl[['object_id', 'date_start', 'date_end']], 
                  how='left', on='object_id')
tr_log['after_peak'] = (tr_log.date_start <= tr_log.date) & (tr_log.date <= tr_log.date_end) 

# TODO: photoz bin

# remove photoz 2.0
tr_log = tr_log[tr_log['hostgal_photoz']>2.0]
"""
In [14]: tr_log.drop_duplicates(['object_id', 'target']).target.value_counts()
Out[14]: 
90    92
42    89
62    38
52    10
Name: target, dtype: int64

"""
template_log = tr_log.copy()

# used oid for template
oid_target = {}
for k,v in tr_log[['object_id' , 'target']].values:
    oid_target[k] = v

def log_to_template(df, target):
    temp = df[(df.target==target) & (df.after_peak==True)]
    temp.mjd -= temp.groupby('object_id').mjd.transform('min')
    temp.flux = temp.flux.
#    temp.flux = temp.flux.clip(0.00001)
    temp.flux /= temp.groupby('object_id').flux.transform('max')
    temp.date = temp.mjd.astype(int)
    temp = pd.pivot_table(temp, index=['date'], columns=['passband'], 
                          values=['flux'], aggfunc='mean')
    
    temp.columns = pd.Index([f'pb{e[1]}' for e in temp.columns.tolist()])
    
    return temp

def lc_template_loo(oid):
    df = template_log[template_log.object_id != oid].reset_index(drop=True)
    
    idxmax = df.groupby('object_id').flux.idxmax()
    tbl = df.iloc[idxmax].reset_index(drop=True)
    tbl['date_start'] = tbl['date']
    tbl['date_end'] = tbl['date'] + 60
    df = pd.merge(df, tbl[['object_id', 'date_start', 'date_end']], how='left', on='object_id')
    df['after_peak'] = (df.date_start <= df.date) & (df.date <= df.date_end) 
    
    template42 = log_to_template(df, 42)
    template52 = log_to_template(df, 52)
    template62 = log_to_template(df, 62)
    template67 = log_to_template(df, 67)
    template90 = log_to_template(df, 90)
    
    return template42, template52, template62, template67, template90

template42 = log_to_template(template_log, 42)
template52 = log_to_template(template_log, 52)
template62 = log_to_template(template_log, 62)
template67 = log_to_template(template_log, 67)
template90 = log_to_template(template_log, 90)


def train_feature(oid):
    if oid in oid_target:
        template42, template52, template62, template67, template90 = lc_template_loo(oid)
    
    
    return


def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    # -178 ~ date ~ +178
    idxmax = df.groupby('object_id').flux.idxmax()
    base = df.iloc[idxmax][['object_id', 'date']]
    li = [base]
    for i in range(178):
        i += 1
        lag  = base.copy()
        lead = base.copy()
        lag['date']  -= i
        lead['date'] += i
        li.append(lag)
        li.append(lead)
    
    keep = pd.concat(li)
    
    df = pd.merge(keep, df, on=['object_id', 'date'], how='inner')
    
    pt = pd.pivot_table(df, index=['object_id'], 
                        aggfunc=num_aggregations)
    
    pt.columns = pd.Index([f'{e[0]}_{e[1]}' for e in pt.columns.tolist()])
    
    # std / mean
    col_std = [c for c in pt.columns if c.endswith('_std')]
    for c in col_std:
        pt[f'{c}-d-mean'] = pt[c]/pt[c.replace('_std', '_mean')]
    
    # max / min, max - min
    col_max = [c for c in pt.columns if c.endswith('_max')]
    for c in col_max:
        pt[f'{c}-d-min'] = pt[c]/pt[c.replace('_max', '_min')]
        pt[f'{c}-m-min'] = pt[c]-pt[c.replace('_max', '_min')]
    
    # q75 - q25, q90 - q10
    col = [c for c in pt.columns if c.endswith('_q75')]
    for c in col:
        x = c.replace('_q75', '')
        pt[f'{x}_q75-m-q25'] = pt[c] - pt[c.replace('_q75', '_q25')]
        pt[f'{x}_q90-m-q10'] = pt[c.replace('_q75', '_q90')] - pt[c.replace('_q75', '_q10')]
    
    
    if drop_oid:
        pt.reset_index(drop=True, inplace=True)
    else:
        pt.reset_index(inplace=True)
    pt.add_prefix(PREF+'_').to_pickle(output_path)
    
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

    aggregate(, f'../data/train_{PREF}.pkl')
    
    # test
    if utils.GENERATE_TEST:
        imp = pd.read_csv(utils.IMP_FILE).head(utils.GENERATE_FEATURE_SIZE)
        usecols = imp[imp.feature.str.startswith(f'{PREF}')][imp.gain>0].feature.tolist()
        
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
        utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
        utils.save_test_features(df[usecols])
        os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)




