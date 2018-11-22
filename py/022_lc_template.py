#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:29:46 2018

@author: kazuki.onodera



parameters:
    classes: [42, 52, 62, 67, 90]
    days: [10, 20, 30]
    aggfunc: [mean, median]
    detected: [0, 1]
    specz
    

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

PREF = 'f022'


os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

DAYS = 10


# =============================================================================
# def
# =============================================================================

class_SN = [42, 52, 62, 67, 90]

tr = pd.read_pickle('../data/train.pkl')


tr_log = pd.read_pickle('../data/train_log.pkl')
tr_log = pd.merge(tr_log, tr[['object_id', 'hostgal_photoz', 'target']], 
                  on='object_id', how='left')
tr_log = tr_log[(tr_log.target.isin(class_SN)) & (tr_log.flux>0)].reset_index(drop=True) # remove flux<0


# after peak for 60days
idxmax = tr_log.groupby('object_id').flux.idxmax()
tbl = tr_log.iloc[idxmax].reset_index(drop=True)
tbl['date_start'] = tbl['date']
tbl['date_end'] = tbl['date'] + DAYS
tr_log = pd.merge(tr_log, tbl[['object_id', 'date_start', 'date_end']], 
                  how='left', on='object_id')
tr_log['after_peak'] = (tr_log.date_start <= tr_log.date) & (tr_log.date <= tr_log.date_end) 

# TODO: specz bin

# remove specz 2.0
#tr_log = tr_log[tr_log['hostgal_specz']>2.0]
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

target_oids = {}
for t in class_SN:
    target_oids[t] = tr[tr.target==t].object_id.tolist()



comb = [('pb0', 'pb1'),
         ('pb0', 'pb2'),
         ('pb0', 'pb3'),
         ('pb0', 'pb4'),
         ('pb0', 'pb5'),
         ('pb1', 'pb2'),
         ('pb1', 'pb3'),
         ('pb1', 'pb4'),
         ('pb1', 'pb5'),
         ('pb2', 'pb3'),
         ('pb2', 'pb4'),
         ('pb2', 'pb5'),
         ('pb3', 'pb4'),
         ('pb3', 'pb5'),
         ('pb4', 'pb5')]


def log_to_template(df, target):
    temp = df[(df.target==target) & (df.after_peak==True)].reset_index(drop=True)
    temp.mjd -= temp.groupby('object_id').mjd.transform('min')
    temp.flux /= temp.groupby('object_id').flux.transform('max')
    temp.date = temp.mjd.astype(int)
    temp = pd.pivot_table(temp, index=['date'], columns=['passband'], 
                          values=['flux'], aggfunc='mean')
    
    temp.columns = pd.Index([f'pb{e[1]}' for e in temp.columns.tolist()])
    
    # compare passband
    for c1,c2 in comb:
        temp[f'{c1}-d-{c2}'] = temp[c1] / temp[c2]
    return temp

np.random.seed(71)
def lc_template_l3o(oid):
    """
    leave (three * other class) out
    """
#    t = oid_target[oid]
    oids = []
    for c in class_SN:
        oids += list(np.random.choice(target_oids[c], size=3, replace=False))
    if oid in oids:
        pass
    else:
        oids.append(oid)
    df = template_log[~template_log.object_id.isin(oids)].reset_index(drop=True)
    
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



def multi_train(oid):
#    if oid in oid_target:
    template42_, template52_, template62_, template67_, template90_ = lc_template_l3o(oid)
#    else:
#        template42_, template52_, template62_, template67_, template90_ = \
#        template42.copy(), template52.copy(), template62.copy(), template67.copy(), template90.copy()
    
    df = tr_pt[tr_pt.object_id==oid]
    feature = pd.DataFrame(index=df.index)
    feature['object_id'] = df['object_id']
    li = [feature]
    li.append( (df.iloc[:,1:] / template42_).add_prefix('c42_') )
    li.append( (df.iloc[:,1:] / template52_).add_prefix('c52_') )
    li.append( (df.iloc[:,1:] / template62_).add_prefix('c62_') )
    li.append( (df.iloc[:,1:] / template67_).add_prefix('c67_') )
    li.append( (df.iloc[:,1:] / template90_).add_prefix('c90_') )
    feature = pd.concat(li, axis=1, join='inner')
    
    return feature

#multi_train_oid(613)
#multi_train_oid(730)



def multi_test(args):
    input_path, output_path = args
    
    te_log = pd.read_pickle(input_path)
    idxmax = te_log.groupby('object_id').flux.idxmax()
    base = te_log.iloc[idxmax][['object_id', 'date']]
    li = [base]
    for i in range(10):
        i += 1
        lead = base.copy()
        lead['date'] += i
        li.append(lead)
    
    keep = pd.concat(li)
    
    te_log = pd.merge(keep, te_log, on=['object_id', 'date'], how='inner')
    te_log.mjd -= te_log.groupby('object_id').mjd.transform('min')
    te_log.date = te_log.mjd.astype(int)
    te_log.flux /= te_log.groupby('object_id').flux.transform('max')
    
    te_pt = pd.pivot_table(te_log, index=['object_id', 'date'], 
                           columns=['passband'], values=['flux'], aggfunc='mean')
    te_pt.columns = pd.Index([f'pb{e[1]}' for e in te_pt.columns.tolist()])
    
    # compare passband
    for c1,c2 in comb:
        te_pt[f'{c1}-d-{c2}'] = te_pt[c1] / te_pt[c2]
    te_pt.reset_index(inplace=True)
    te_pt.set_index('date', inplace=True)
    
    # TODO: write
    feature = te_pt[['object_id']]
    feature['date'] = te_pt.index
    feature.reset_index(drop=True, inplace=True)
    li = [feature]
    li.append( (te_pt.iloc[:,1:] / template42).add_prefix('c42_').reset_index(drop=True) )
    li.append( (te_pt.iloc[:,1:] / template52).add_prefix('c52_').reset_index(drop=True) )
    li.append( (te_pt.iloc[:,1:] / template62).add_prefix('c62_').reset_index(drop=True) )
    li.append( (te_pt.iloc[:,1:] / template67).add_prefix('c67_').reset_index(drop=True) )
    li.append( (te_pt.iloc[:,1:] / template90).add_prefix('c90_').reset_index(drop=True) )
    feature = pd.concat(li, axis=1, join='inner')
    
    pt = pd.pivot_table(feature, index=['object_id'], 
                        columns=['date'])
    pt.columns = pd.Index([f'{e[0]}_d{e[1]}' for e in pt.columns.tolist()])
    pt.reset_index(inplace=True) # keep oid
    pt.add_prefix(PREF+'_').to_pickle(output_path)
    
    return


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    usecols = None
    
    # =============================================================================
    # # train
    # =============================================================================
    tr_log = pd.read_pickle('../data/train_log.pkl')
    idxmax = tr_log.groupby('object_id').flux.idxmax()
    base = tr_log.iloc[idxmax][['object_id', 'date']]
    li = [base]
    for i in range(10):
        i += 1
        lead = base.copy()
        lead['date'] += i
        li.append(lead)
    
    keep = pd.concat(li)
    
    tr_log = pd.merge(keep, tr_log, on=['object_id', 'date'], how='inner')
    tr_log.mjd -= tr_log.groupby('object_id').mjd.transform('min')
    tr_log.date = tr_log.mjd.astype(int)
    tr_log.flux /= tr_log.groupby('object_id').flux.transform('max')
    
    tr_pt = pd.pivot_table(tr_log, index=['object_id', 'date'], 
                           columns=['passband'], values=['flux'], aggfunc='mean')
    tr_pt.columns = pd.Index([f'pb{e[1]}' for e in tr_pt.columns.tolist()])
    
    # compare passband
    for c1,c2 in comb:
        tr_pt[f'{c1}-d-{c2}'] = tr_pt[c1] / tr_pt[c2]
    tr_pt.reset_index(inplace=True)
    tr_pt.set_index('date', inplace=True)
    
    pool = Pool( cpu_count() )
    callback = pool.map(multi_train, tr.object_id.tolist())
    pool.close()
    
    callback = pd.concat(callback).reset_index()
    
    pt = pd.pivot_table(callback, index=['object_id'], 
                        columns=['date']).reset_index(drop=True)
    
    pt.columns = pd.Index([f'{e[0]}_d{e[1]}' for e in pt.columns.tolist()])
    pt.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')
    
    # test
    if utils.GENERATE_TEST:
        imp = pd.read_csv(utils.IMP_FILE).head(utils.GENERATE_FEATURE_SIZE)
        usecols = imp[imp.feature.str.startswith(f'{PREF}')][imp.gain>0].feature.tolist()
        
        os.system(f'rm ../data/tmp_{PREF}*')
        argss = []
        for i,file in enumerate(utils.TEST_LOGS):
            argss.append([file, f'../data/tmp_{PREF}{i}.pkl'])
        pool = Pool( cpu_count() )
        pool.map(multi_test, argss)
        pool.close()
        df = pd.concat([pd.read_pickle(f) for f in glob(f'../data/tmp_{PREF}*')], 
                        ignore_index=True)
        df.sort_values(f'{PREF}_object_id', inplace=True)
        df.reset_index(drop=True, inplace=True)
        utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
        utils.save_test_features(df[usecols])
        os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)




