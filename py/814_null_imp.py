#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:42:55 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm

import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count

import utils, utils_metric
#utils.start(__file__)
#==============================================================================



SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5

LOOP = 1

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.7,
         'max_depth': 3,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 100,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.7,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         
         'seed': SEED
         }

USE_FEATURES = 1000

# =============================================================================
# load
# =============================================================================
COL = pd.read_csv(utils.IMP_FILE).head(USE_FEATURES ).feature.tolist()


PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=10)
               ], axis=1)[COL]
y = utils.load_target().target

#X.drop(DROP, axis=1, inplace=True)

target_dict = {}
target_dict_r = {}
for i,e in enumerate(y.sort_values().unique()):
    target_dict[e] = i
    target_dict_r[i] = e

y = y.replace(target_dict)

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()


# =============================================================================
# 
# =============================================================================




def get_imp(shuffle=True):
    
    if shuffle:
        dtrain = lgb.Dataset(X, y.sample(frac=1).values, free_raw_data=False)
        gc.collect()
    else:
        dtrain = lgb.Dataset(X, y.values, free_raw_data=False)
        gc.collect()
    
    model_all = []
    nround_mean = 0
    wloss_list = []
    for i in range(LOOP):
        gc.collect()
        param['seed'] = np.random.randint(9999)
        ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                             fobj=utils_metric.wloss_objective, 
                             feval=utils_metric.wloss_metric,
                             early_stopping_rounds=100, verbose_eval=50,
                             seed=SEED+i)
        model_all += models
        nround_mean += len(ret['multi_logloss-mean'])
        wloss_list.append( ret['wloss-mean'][-1] )
    
    imp = ex.getImp(model_all)
    imp['split'] /= imp['split'].max()
    imp['gain'] /= imp['gain'].max()
    imp['total'] = imp['split'] + imp['gain']
    
    imp.sort_values('total', ascending=False, inplace=True)
    imp.reset_index(drop=True, inplace=True)
    
    return imp.set_index('feature')

imp = get_imp(False)
imp['null_imp'] = 0

for i in range(100):
    imp['null_imp'] += get_imp().total

imp.to_csv(f'LOG/imp_{__file__}.csv')

#==============================================================================
utils.end(__file__)
utils.stop_instance()






