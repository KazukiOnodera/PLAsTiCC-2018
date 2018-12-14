#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:31:03 2018

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


DROP = ['f001_hostgal_specz', 'f001_distmod',]# 'f701_hostgal_specz'] # 

SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5

LOOP = 5

param = {
         'objective': 'multiclass',
         'num_class': 9,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.5,
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


# =============================================================================
# load
# =============================================================================
is_gal = pd.read_pickle('../data/tr_is_gal.pkl')

files_tr = sorted(glob('../data/train_f*.pkl'))
[print(i,f) for i,f in enumerate(files_tr)]

X = pd.read_pickle('../FROM_MYTEAM/full_train_v103_074_thresh_6.pkl.gz')
y = utils.load_target().target

y = y[~is_gal]

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
# cv2
# =============================================================================

param['learning_rate'] = 0.01
dtrain = lgb.Dataset(X, y, free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
y_preds = []
for i in range(5):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models
    
    y_pred = ex.eval_oob(X, y.values, models, SEED+i, stratified=True, shuffle=True, 
                         n_class=True)
    y_preds.append(y_pred)
    
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['multi_logloss-mean'][-1] )



result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
utils.send_line(result)

imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

print(imp.head(200).feature.map(lambda x: x.split('_')[0]).value_counts())


#==============================================================================
utils.end(__file__)
#utils.stop_instance()



