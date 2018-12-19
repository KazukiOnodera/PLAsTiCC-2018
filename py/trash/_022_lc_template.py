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
    date_from: 10 days before from peak
    template augment: True
    train, test augment: True
    

"""



import numpy as np
import pandas as pd
import os
from glob import glob
from multiprocessing import cpu_count, Pool
from itertools import combinations

import sys
argvs = sys.argv

import utils

PREF = 'f022'


os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')


DAYS_FROM = 10
DAYS_TO = 10

DATE_AUGMENT = 2

class_SN = [42, 52, 62, 67, 90]

tr = pd.read_pickle('../data/train.pkl')

# =============================================================================
# template
# =============================================================================
tr_log = pd.read_pickle('../data/train_log.pkl')
tr_log = pd.merge(tr_log, tr[['object_id', 'hostgal_photoz', 'target']], 
                  on='object_id', how='left')
tr_log = tr_log[(tr_log.target.isin(class_SN))].reset_index(drop=True)


# -DAYS_FROM ~ peak + DAYS_TO
idxmax = tr_log.groupby('object_id').flux.idxmax()
base = tr_log.iloc[idxmax][['object_id', 'date']]
li = []
for i in range(-DAYS_FROM, 0):
    tmp = base.copy()
    tmp['date'] += i
    li.append(tmp)

lag = pd.concat(li)
lag = pd.merge(lag, tr_log, on=['object_id', 'date'], how='left')
lag = lag.sort_values(['object_id', 'date']).reset_index(drop=True)

li = []
for i in range(0, DAYS_TO):
    tmp = base.copy()
    tmp['date'] += i
    li.append(tmp)

lead = pd.concat(li)
lead = pd.merge(lead, tr_log, on=['object_id', 'date'], how='left')
lead = lead[lead.object_id.isin(lag.object_id)].sort_values(['object_id', 'date']).reset_index(drop=True)

tr_log = pd.concat([lag, lead], ignore_index=True).sort_values(['object_id', 'date']).reset_index(drop=True)

# TODO: specz bin

# remove specz 2.0
#tr_log = tr_log[tr_log['hostgal_specz']>2.0]


template_log = tr_log.copy()

# used oid for template
oid_target = {}
for k,v in tr_log[['object_id' , 'target']].values:
    oid_target[k] = v

target_oids = {}
for t in class_SN:
    target_oids[t] = tr_log[tr_log.target==t].object_id.unique().tolist()

# =============================================================================
# def
# =============================================================================
def norm_flux_date(df):
#    df.flux -= df.groupby(['object_id']).flux.transform('min')
    df.flux /= df.groupby('object_id').flux.transform('max')
    df.date -= df.groupby('object_id').date.transform('min')

norm_flux_date(template_log)

# augment
def augment(df, n):
    if n > 0:
        li = []
        for i in range(1, n+1):
            tmp = df.copy()
            tmp['date'] += i
            li.append(tmp)
            tmp = df.copy()
            tmp['date'] -= i
            li.append(tmp)
        tmp = pd.concat(li)
        tmp = tmp[tmp.date.between(0, 19)]
        df = pd.concat([df, tmp], ignore_index=True)
    return df

template_log = augment(template_log, DATE_AUGMENT)


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
    temp = df[(df.target==target)].reset_index(drop=True)
    temp = pd.pivot_table(temp, index=['date'], columns=['passband'], 
                          values=['flux'], aggfunc='mean')
    
    temp.columns = pd.Index([f'pb{int(e[1])}' for e in temp.columns.tolist()])
    
    # compare passband
    for c1,c2 in comb:
        temp[f'{c1}-d-{c2}'] = temp[c1] / temp[c2]
    temp['object_id'] = 1
    return temp


def lc_template_l3o(oid):
    """
    leave (three * other class) out
    """
    np.random.seed(oid)
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

template42 = log_to_template(template_log, 42) # index is date
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
    li = []
    li.append( df.reset_index().set_index(['object_id', 'date']) ) 
    li.append( (df / template42_).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c42_') )
    li.append( (df / template52_).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c52_') )
    li.append( (df / template62_).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c62_') )
    li.append( (df / template67_).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c67_') )
    li.append( (df / template90_).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c90_') )
    feature = pd.concat(li, axis=1, join='inner')
    
    return feature

#multi_train_oid(613)
#multi_train_oid(730)



def multi_test(args):
    """
    input_path, output_path = '../data/test_log01.pkl', '../data/tmp_f0221.pkl'
    """
    input_path, output_path = args
    
    te_log = pd.read_pickle(input_path)
    idxmax = te_log.groupby('object_id').flux.idxmax()
    base = te_log.iloc[idxmax][['object_id', 'date']]
    li = []
    for i in range(-DAYS_FROM, DAYS_TO):
        tmp = base.copy()
        tmp['date'] += i
        li.append(tmp)
    
    keep = pd.concat(li)
    
    te_log = pd.merge(keep, te_log, on=['object_id', 'date'], how='left')
    te_log = te_log.sort_values(['object_id', 'date']).reset_index(drop=True)
    norm_flux_date(te_log)
    te_log = augment(te_log, DATE_AUGMENT)
    
    te_pt = pd.pivot_table(te_log, index=['object_id', 'date'], 
                           columns=['passband'], values=['flux'], aggfunc='mean')
    te_pt.columns = pd.Index([f'pb{int(e[1])}' for e in te_pt.columns.tolist()])
    
    # compare passband
    for c1,c2 in comb:
        te_pt[f'{c1}-d-{c2}'] = te_pt[c1] / te_pt[c2]
    te_pt.reset_index(inplace=True)
    te_pt.set_index('date', inplace=True)
    
    li = []
    li.append( te_pt.reset_index().set_index(['object_id', 'date']) ) # keep date
    li.append( (te_pt / template42).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c42_') )
    li.append( (te_pt / template52).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c52_') )
    li.append( (te_pt / template62).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c62_') )
    li.append( (te_pt / template67).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c67_') )
    li.append( (te_pt / template90).dropna(how='all').reset_index().set_index(['object_id', 'date']).add_prefix('c90_') )
    feature = pd.concat(li, axis=1, join='inner').reset_index()
    
    pt = pd.pivot_table(feature, index=['object_id'], 
                        columns=['date']) # unique oid, date
    pt.columns = pd.Index([f'{e[0]}_d{e[1]}' for e in pt.columns.tolist()])
    pt.reset_index(inplace=True) # keep oid
    pt.add_prefix(PREF+'_').to_pickle(output_path)
    
    return


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    
    # =============================================================================
    # # train
    # =============================================================================
    tr_log = pd.read_pickle('../data/train_log.pkl')
    idxmax = tr_log.groupby('object_id').flux.idxmax()
    base = tr_log.iloc[idxmax][['object_id', 'date']]
    li = []
    for i in range(-DAYS_FROM, DAYS_TO):
        tmp = base.copy()
        tmp['date'] += i
        li.append(tmp)
    
    keep = pd.concat(li)
    
    tr_log = pd.merge(keep, tr_log, on=['object_id', 'date'], how='left')
    tr_log = tr_log.sort_values(['object_id', 'date']).reset_index(drop=True)
    norm_flux_date(tr_log)
    tr_log = augment(tr_log, DATE_AUGMENT)
    
    tr_pt = pd.pivot_table(tr_log, index=['object_id', 'date'], 
                           columns=['passband'], values=['flux'], aggfunc='mean')
    tr_pt.columns = pd.Index([f'pb{int(e[1])}' for e in tr_pt.columns.tolist()])
    
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
        del df[f'{PREF}_object_id']
        utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
        utils.save_test_features(df[usecols])
        os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)




