#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:16:05 2018

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
         
         'colsample_bytree': 0.5,
         'subsample': 0.7,
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
COL = pd.read_csv('LOG/imp_801_cv.py-2.csv').head(USE_FEATURES ).feature.tolist()


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
# cv
# =============================================================================

model_all = []
nround_mean = 0
wloss_list = []
y_preds = []

dtrain = lgb.Dataset(X[COL], y.values, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()


for i in range(LOOP):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
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


for i,y_pred in enumerate(y_preds):
    y_pred = utils_metric.softmax(y_pred.astype(float).values)
    if i==0:
        y_preds_ = y_pred
    else:
        y_preds_ += y_pred

y_preds_ /= len(y_preds)

# =============================================================================
# 
# =============================================================================

utils_metric.multi_weighted_logloss(y, y_preds_)

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
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=1)
    
    return pd.Series( -y_log_ones )


multi_weighted_logloss(y, y_preds_)

loss = multi_weighted_logloss(y, y_preds_)
loss.name = 'loss'

sub = pd.concat([pd.read_pickle('../data/train.pkl').object_id, y, loss], 
                 axis=1)


sub['loss_mean'] = sub.groupby('target').loss.transform('mean')

sub['loss_diff'] = (sub.loss - sub.loss_mean).abs()

sub.sort_values('loss_diff', ascending=False, inplace=True)
#sub.reset_index(inplace=True, drop=True)

"""

In [72]: sub.index[:100]
Out[72]: 
Int64Index([3446, 1601, 6684, 6969, 2859, 6615, 7312, 1707, 6337, 6084, 2702,
            3690,  774, 7763, 7694, 5530,  376, 4465, 4114, 3795, 7128, 6557,
            3611, 2983, 3369, 3037, 2881, 2132, 2575, 2611, 7654, 3679, 5678,
            4590, 5981, 3426, 4151, 2393, 6530, 2449, 6665, 2784, 2832, 3998,
            5119, 7384, 5876, 5566, 4142, 4695, 6360, 3314, 4218, 7111, 2834,
            3287, 3906, 3282, 3267, 5748, 6481, 4423, 3549, 2250, 4333, 6370,
            4708, 5170, 7283, 3645, 7559, 3825, 4140, 3194, 5369, 2656, 6975,
            4740, 4387, 6374, 3118, 2638, 6249, 2452, 6747, 2224, 3917, 5944,
            4070, 5098, 3271, 4645, 3493, 5696,  333, 6723, 6749, 1064, 2969,
            7357],
           dtype='int64')


"""


#==============================================================================
utils.end(__file__)
utils.stop_instance()



