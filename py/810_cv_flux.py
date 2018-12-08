#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 03:32:52 2018

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
print('SEED:', SEED)

DROP = ['f001_hostgal_specz', 'f001_distmod',]# 'f701_hostgal_specz'] # 

#DROP = []

NFOLD = 5

LOOP = 1

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
#         'learning_rate': 0.01,
         'max_depth': 3,
#         'num_leaves': 63,
         'max_bin': 127,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.7,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }

taguchi_param = {
                'objective': 'multiclass',
                'num_class': 14,
                'nthread': cpu_count(),
                'learning_rate': 0.4,
                'max_depth': 3,
                'subsample': .9,
                'colsample_bytree': .7,
                'reg_alpha': .01,
                'reg_lambda': .01,
                'min_split_gain': 0.01,
                'min_child_weight': 200,
                'verbose': -1,
                
                'max_bin': 20,
        #        'min_data_in_leaf': 30,
        #        'bagging_fraction',
        #        'bagging_freq',
            }

#param = taguchi_param

# =============================================================================
# load
# =============================================================================

files_tr = sorted(glob('../data/train_f*.pkl'))
[print(i,f) for i,f in enumerate(files_tr)]

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
y = utils.load_target().target

X.drop(DROP, axis=1, inplace=True)

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
# select flux features
# =============================================================================


col_flux = [c for c in X.columns if 'flux' in c]
col_flux1 = [c for c in col_flux if 'flux_norm1' in c]
col_flux2 = [c for c in col_flux if 'flux_norm2' in c]
col_flux3 = [c for c in col_flux if 'flux_norm3' in c]
col_fluxr = [c for c in col_flux if 'flux_ratio_sq' in c]
col_fluxb = [c for c in col_flux if 'flux_by_flux_ratio_sq' in c]


col_flux_org = list( set(col_flux) - set(col_flux1) - set(col_flux2) - set(col_flux3) \
                    - set(col_fluxr)- set(col_fluxb))

col_di = {
        'org': col_flux_org,
        'flux1': col_flux1,
        'flux2': col_flux2,
        'flux3': col_flux3,
        'flux1': col_flux1,
        'r': col_fluxr,
        'b': col_fluxb,
          }

def cv(key):
    """
    key = 'org'
    """
    print(key)
    col_drop = list( set(col_flux) - set(col_di[key]) )
    
    # =============================================================================
    # cv1
    # =============================================================================
    param['learning_rate'] = 0.1
    dtrain = lgb.Dataset(X.drop(col_drop, axis=1), y, #categorical_feature=CAT, 
                         free_raw_data=False)
    gc.collect()
    
    model_all = []
    nround_mean = 0
    wloss_list = []
    for i in range(LOOP):
        gc.collect()
        param['seed'] = np.random.randint(9999)
        ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                             feval=utils_metric.lgb_multi_weighted_logloss,
                             early_stopping_rounds=100, verbose_eval=50,
                             seed=SEED)
        model_all += models
        nround_mean += len(ret['multi_logloss-mean'])
        wloss_list.append( ret['wloss-mean'][-1] )
    
    nround_mean = int((nround_mean/LOOP) * 1.3)
    
    result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
    utils.send_line(result)
    
    imp = ex.getImp(model_all)
    imp['split'] /= imp['split'].max()
    imp['gain'] /= imp['gain'].max()
    imp['total'] = imp['split'] + imp['gain']
    
    imp.sort_values('total', ascending=False, inplace=True)
    imp.reset_index(drop=True, inplace=True)
    
    print(imp.head(100).feature.map(lambda x: x.split('_')[0]).value_counts())
        
    """
    
    __file__ = '810_cv_flux.py'
    imp = pd.read_csv(f'LOG/imp_{__file__}-1.csv')
    
    """
    
    
    
    # =============================================================================
    # cv2
    # =============================================================================
    COL = imp.feature.tolist()[:3000]
    
    param['learning_rate'] = 0.5
    dtrain = lgb.Dataset(X[COL], y, #categorical_feature=CAT, 
                         free_raw_data=False)
    gc.collect()
    
    model_all = []
    nround_mean = 0
    wloss_list = []
    for i in range(1):
        gc.collect()
        param['seed'] = np.random.randint(9999)
        ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                                fobj=utils_metric.wloss_objective, 
                                feval=utils_metric.wloss_metric,
                             early_stopping_rounds=100, verbose_eval=50,
                             seed=SEED)
        model_all += models
        nround_mean += len(ret['multi_logloss-mean'])
        wloss_list.append( ret['wloss-mean'][-1] )
    
    #nround_mean = int((nround_mean/LOOP) * 1.3)
    
    result = f"CV {key} wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
    utils.send_line(result)
    
    imp = ex.getImp(model_all)
    imp['split'] /= imp['split'].max()
    imp['gain'] /= imp['gain'].max()
    imp['total'] = imp['split'] + imp['gain']
    
    imp.sort_values('total', ascending=False, inplace=True)
    imp.reset_index(drop=True, inplace=True)
    
    print(imp.head(200).feature.map(lambda x: x.split('_')[0]).value_counts())
    
    #imp.to_csv(f'LOG/imp_{__file__}-{key}.csv', index=False)
    
    
    
    # =============================================================================
    # estimate feature size
    # =============================================================================
    print('estimate feature size')
    param['learning_rate'] = 0.5
    
    COL = imp.feature.tolist()
    best_score = 9999
    best_N = 0
    
    for i in np.arange(100, 500, 50):
        print(f'\n==== feature size: {i} ====')
        
        dtrain = lgb.Dataset(X[COL[:i]], y, #categorical_feature=CAT, 
                             free_raw_data=False)
        gc.collect()
        param['seed'] = np.random.randint(9999)
        ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                             fobj=utils_metric.wloss_objective, 
                             feval=utils_metric.wloss_metric,
                             early_stopping_rounds=100, verbose_eval=50,
                             seed=SEED)
        
        score = ret['wloss-mean'][-1]
#        utils.send_line(f"feature size: {i}    wloss-mean: {score}")
        
        if score < best_score:
            best_score = score
            best_N = i
    
    print('best_N', best_N)

# =============================================================================
# 
# =============================================================================

[cv(k) for k in col_di]


#==============================================================================
utils.end(__file__)
utils.stop_instance()


