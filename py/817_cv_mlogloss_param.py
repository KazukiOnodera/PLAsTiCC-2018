#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:44:48 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm

import optuna

import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count


import utils, utils_metric
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)


NFOLD = 5

LOOP = 2


param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.1,
#         'learning_rate': 0.05,
         'max_depth': 3,
         'num_leaves': 63,
         'max_bin': 127,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 100,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         
         'seed': SEED
         }

USE_FEATURES = 100

# =============================================================================
# load
# =============================================================================

COL = pd.read_csv('LOG/imp_used_1213-1.csv').head( USE_FEATURES ).feature.tolist()

PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

files_te = [f'../feature/test_{c}.pkl' for c in COL]
sw = False
for i in files_te:
    if os.path.exists(i)==False:
        print(i)
        sw = True
if sw:
    raise Exception()

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)[COL]
y = utils.load_target().target


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

dtrain = lgb.Dataset(X, y, free_raw_data=False)
gc.collect()

# =============================================================================
# optuna
# =============================================================================


#def objective(trial):
#    param['max_depth'] = trial.suggest_int('max_depth', 3, 6)
#    param['subsample'] = trial.suggest_discrete_uniform('subsample', 0.5, 1.0, 0.1)
#    param['colsample_bytree'] = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1.0, 0.1)
#    param['reg_alpha'] = trial.suggest_discrete_uniform('reg_alpha', 0.01, 0.5, 0.01)
#    param['reg_lambda'] = trial.suggest_discrete_uniform('reg_lambda', 0.01, 0.5, 0.01)
#    param['min_split_gain'] = trial.suggest_discrete_uniform('min_split_gain', 0.01, 0.1, 0.01)
#    param['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 300)
#    param['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 300)
#    param['max_bin'] = trial.suggest_int('max_bin', 15, 127)
#    
#    model_all = []
#    nround_mean = 0
#    wloss_list = []
#    for i in range(LOOP):
#        gc.collect()
#        param['seed'] = np.random.randint(9999)
#        ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD,
#                             early_stopping_rounds=100, verbose_eval=100,
#                             seed=SEED+i)
#        model_all += models
#        nround_mean += len(ret['multi_logloss-mean'])
#        wloss_list.append( ret['multi_logloss-mean'][-1] )
#    
#    return np.mean(wloss_list)
#
#
## optuna
#study = optuna.create_study()
#study.optimize(objective, n_trials=9999)
#
## 最適解
#print(study.best_params)
#print(study.best_value)
#print(study.best_trial)

# =============================================================================
# grid
# =============================================================================


from itertools import product

param_list = []

# colsample_bytree & subsample
for i,j in product(np.arange(0.1, 1, 0.2), np.arange(0.5, 1, 0.2)):
    param_ = param.copy()
    param_['colsample_bytree'] = round(i, 2)
    param_['subsample'] = round(j, 2)
    param_list.append(param_)



model_all = []
nround_mean = 0
wloss_list = []
y_preds = []


for param_ in param_list:
    gc.collect()
    print(f"\ncolsample_bytree: {param_['colsample_bytree']}    subsample: {param_['subsample']}")
    ret, models = lgb.cv(param_, dtrain, 99999, nfold=NFOLD, 
                         fobj=utils_metric.wloss_objective, 
                         feval=utils_metric.wloss_metric,
                         early_stopping_rounds=100, verbose_eval=150,
                         seed=SEED)
    y_pred = ex.eval_oob(X, y.values, models, SEED, stratified=True, shuffle=True, 
                         n_class=y.unique().shape[0])
    y_preds.append(y_pred)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['multi_logloss-mean'][-1] )







#==============================================================================
utils.end(__file__)
utils.stop_instance()


