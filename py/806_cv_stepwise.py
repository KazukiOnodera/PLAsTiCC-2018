#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 19:04:06 2018

@author: Kazuki
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

LOOP = 5

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.5,
         'max_depth': 3,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 100,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
#         'colsample_bytree': 0.5,
#         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         
         'seed': SEED
         }

USE_FEATURES = 500

# =============================================================================
# load
# =============================================================================
COL = pd.read_csv(utils.IMP_FILE_BEST).head(USE_FEATURES ).feature.tolist()


PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
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
# search param
# =============================================================================

features_search = X.columns.tolist()
features_curr = features_search[:10]

# =============================================================================
# cv
# =============================================================================

model_all = []
nround_mean = 0
wloss_list = []
y_preds = []

dtrain = lgb.Dataset(X, y.values, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()
    
for param_ in param_list:
    gc.collect()
    print(f"\ncolsample_bytree: {param_['colsample_bytree']}    subsample: {param_['subsample']}")
    ret, models = lgb.cv(param_, dtrain, 99999, nfold=NFOLD, 
                         fobj=utils_metric.wloss_objective, 
                         feval=utils_metric.wloss_metric,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    y_pred = ex.eval_oob(X[COL], y.values, models, SEED, stratified=True, shuffle=True, 
                         n_class=y.unique().shape[0])
    y_preds.append(y_pred)
    model_all += models
    nround_mean += len(ret['wloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

ex.stepwise(param, X, y, features_search, features_curr, best_score=0, send_line=None,
             eval_key='wloss-mean', maximize=False, save_df=None, cv_loop=1,
             num_boost_round=100, 
             folds=None, nfold=5, stratified=True, shuffle=True, metrics=None, 
             fobj=utils_metric.wloss_objective, 
             feval=utils_metric.wloss_metric, 
             init_model=None, feature_name='auto', categorical_feature='auto', 
             esr=None, fpreproc=None, verbose_eval=None, show_stdv=True, 
             seed=0, callbacks=None)


#==============================================================================
utils.end(__file__)
utils.stop_instance()

