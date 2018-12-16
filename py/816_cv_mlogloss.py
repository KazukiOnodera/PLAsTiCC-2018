#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:25:24 2018

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
utils.start(__file__)
#==============================================================================


DROP = ['f001_hostgal_specz', 'f001_distmod',]# 'f701_hostgal_specz'] # 

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

#REMOVE_PREFS = ['f006', 'f012', 'f017', 'f018', 'f020', 'f023', 'f024', 'f025']

REMOVE_PREFS = ['f706', 'f701', 'f703', 'f704', 'f705', 'f707']

# =============================================================================
# load
# =============================================================================

files_tr = sorted(glob('../data/train_f*.pkl'))

li = []
for a in files_tr:
    for b in REMOVE_PREFS:
        if b in a:
            li.append(a)

files_tr = sorted(set(files_tr) - set(li))
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

#CAT = list( set(X.columns)&set(utils_cat.ALL))
#print(f'CAT: {CAT}')

# =============================================================================
# cv1
# =============================================================================
param['learning_rate'] = 0.1
dtrain = lgb.Dataset(X, y, free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
for i in range(2):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED+i)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['multi_logloss-mean'][-1] )

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

imp.to_csv(f'LOG/imp_{__file__}-1.csv', index=False)

"""

__file__ = '816_cv_mlogloss.py'
imp = pd.read_csv(f'LOG/imp_{__file__}-1.csv')

"""

# =============================================================================
# cv2
# =============================================================================
COL = imp.feature.tolist()[:3000]

param['learning_rate'] = 0.1
dtrain = lgb.Dataset(X[COL], y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
for i in range(5):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED+i)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['multi_logloss-mean'][-1] )

#nround_mean = int((nround_mean/LOOP) * 1.3)

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
utils.send_line(result)

imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

print(imp.head(200).feature.map(lambda x: x.split('_')[0]).value_counts())

imp.to_csv(f'LOG/imp_{__file__}-2.csv', index=False)



## =============================================================================
# estimate feature size
# =============================================================================
print('estimate feature size')
param['learning_rate'] = 0.02

COL = imp.feature.tolist()
best_score = 9999
best_N = 0

for i in np.arange(100, 400, 50):
    print(f'\n==== feature size: {i} ====')
    
    dtrain = lgb.Dataset(X[COL[:i]], y, #categorical_feature=CAT, 
                         free_raw_data=False)
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    
    score = ret['multi_logloss-mean'][-1]
    utils.send_line(f"feature size: {i}    multi_logloss-mean: {score}")
    
    if score < best_score:
        best_score = score
        best_N = i

print('best_N', best_N)


# =============================================================================
# cv
# =============================================================================


dtrain = lgb.Dataset(X, y.values, free_raw_data=False)
gc.collect()

y_preds = []
for i in range(2):
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD,
#                         fobj=utils_metric.wloss_objective, 
#                         feval=utils_metric.wloss_metric,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED+i)
    y_pred = ex.eval_oob(X, y.values, models, SEED+i, stratified=True, shuffle=True, 
                         n_class=True)
    y_preds.append(y_pred)

for i,y_pred in enumerate(y_preds):
#    y_pred = utils_metric.softmax(y_pred.astype(float).values)
    if i==0:
        tmp = y_pred
    else:
        tmp += y_pred
tmp /= len(y_preds)
y_preds = tmp.copy().values.astype(float)

weight = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
weight = weight / y_preds.sum(axis=0)



def multi_weighted_logloss(y_true:np.array, y_preds:np.array):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
        
    y_p = y_preds/y_preds.sum(1)[:,None]
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


print('before:', multi_weighted_logloss(y.values, y_preds))
print('after:', multi_weighted_logloss(y.values, y_preds * weight))







#==============================================================================
utils.end(__file__)
#utils.stop_instance()



