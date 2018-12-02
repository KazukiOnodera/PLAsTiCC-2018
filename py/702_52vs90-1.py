#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:46:39 2018

@author: Kazuki

52 vs 90

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


import utils
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)

DROP = ['f001_hostgal_specz', 'f001_distmod',]# 'f701_hostgal_specz'] # 

#DROP = []

NFOLD = 5

LOOP = 1

param = {
         'objective': 'binary',
         'metric': 'auc',
         
         'learning_rate': 0.01,
         'max_depth': 3,
#         'num_leaves': 63,
         'max_bin': 20,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 50,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.7,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }

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
# subset
# =============================================================================
print('subset')

X_ = X[y.isin([target_dict[52], target_dict[90],])]
y_ = y[y.isin([target_dict[52], target_dict[90],])]

y_ = (y_==target_dict[52])*1

print(f'X_.shape {X_.shape}')

X_oid = utils.load_train(['object_id', 'target'])

# =============================================================================
# cv1
# =============================================================================

dtrain = lgb.Dataset(X_, y_, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
for i in range(LOOP):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models
    nround_mean += len(ret['auc-mean'])
    wloss_list.append( ret['auc-mean'][-1] )

nround_mean = int((nround_mean/LOOP) * 1.3)

result = f"CV auc: {np.mean(wloss_list)} + {np.std(wloss_list)}"
utils.send_line(result)

imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

print(imp.head(100).feature.map(lambda x: x.split('_')[0]).value_counts())

#imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

"""

__file__ = '702_52vs90.py'
imp = pd.read_csv(f'LOG/imp_{__file__}-1.csv')

"""



# =============================================================================
# cv2
# =============================================================================
COL = imp.feature.tolist()[:3000]

dtrain = lgb.Dataset(X_[COL], y_, #categorical_feature=CAT, 
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
                         seed=SEED)
    model_all += models
    nround_mean += len(ret['auc-mean'])
    wloss_list.append( ret['auc-mean'][-1] )

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


# =============================================================================
# estimate feature size
# =============================================================================
print('estimate feature size')

COL = imp.feature.tolist()
best_score = 9999
best_N = 0

for i in np.arange(100, 500, 50):
    print(f'\n==== feature size: {i} ====')
    
    dtrain = lgb.Dataset(X_[COL[:i]], y_, #categorical_feature=CAT, 
                         free_raw_data=False)
    gc.collect()
    param['seed'] = np.random.randint(9999)
    
    wloss_list = []
    for j in range(3):
        ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                             early_stopping_rounds=100, verbose_eval=100,
                             seed=SEED+j)
        
        wloss_list.append( ret['auc-mean'][-1] )
    
    result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
    utils.send_line(result)
    score = np.mean(wloss_list)
    
    if score > best_score:
        best_score = score
        best_N = i

print('best_N', best_N)



#==============================================================================
utils.end(__file__)
#utils.stop_instance()




