#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:28:29 2018

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

import utils
#utils.start(__file__)
#==============================================================================

PREF = 'f701'


SEED = np.random.randint(9999)
print('SEED:', SEED)

NFOLD = 4

LOOP = 2


param = {
         'objective': 'regression',
         'metric': 'rmse',
         'learning_rate': 0.01,
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.9,
         'subsample': 0.9,
#         'nthread': 21,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }

label_name = 'f001_hostgal_specz'


feature_size = 1000

# =============================================================================
# load
# =============================================================================

files_tr = sorted(glob('../data/train_f*.pkl'))

X_train = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
y_train = X_train[label_name]
X_train.drop(label_name, axis=1, inplace=True)

if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()


# =============================================================================
# feature selection
# =============================================================================
dtrain = lgb.Dataset(X_train[y_train!=0], y_train[y_train!=0],
                     free_raw_data=False)
gc.collect()

model_all = []
for i in range(LOOP):
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, stratified=False,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models

imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

COL = imp.head(feature_size).feature.tolist()


def read(f):
    df = pd.read_pickle(f)
    col = list( set(df.columns) & set(COL) )
    return df[col]



files_te = sorted(glob('../data/test_f*.pkl'))

X_test = pd.concat([
                read(f) for f in tqdm(files_te, mininterval=60)
               ], axis=1)[COL]
    
# hostgal_photoz==0 -> hostgal_specz=0?
X_test.loc[X_test['f001_hostgal_photoz']==0, label_name] = 0

y_test = X_test[label_name]
X_test.drop(label_name, axis=1, inplace=True)


if X_test.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_test.columns[X_test.columns.duplicated()] }')
print('no dup :) ')
print(f'X_test.shape {X_test.shape}')

gc.collect()

# =============================================================================
# cv2
# =============================================================================

X_train_train, y_train_train = X_train[~y_train.isnull()], y_train[~y_train.isnull()]
X_train_test,  y_train_test  = X_train[y_train.isnull()],  y_train[y_train.isnull()]

X_test_train, y_test_train = X_test[~y_test.isnull()], y_test[~y_test.isnull()]
X_test_test,  y_test_test  = X_test[y_test.isnull()],  y_test[y_test.isnull()]

X = pd.concat([X_train_train, X_test_train])[COL]
y = pd.concat([y_train_train, y_test_train])

del X_train_train, X_test_train
del y_train_train, y_test_train
gc.collect()


# remove 0
X = X[y!=0]
y = y[y!=0]
print(f'X.shape {X.shape}')

dtrain = lgb.Dataset(X, y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
for i in range(LOOP):
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, stratified=False,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models

result = f"CV rmse-mean: {ret['rmse-mean'][-1]} + {ret['rmse-stdv'][-1]}"
print(result)

utils.send_line(result)
imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)



# =============================================================================
# save
# =============================================================================
train_ind = y_train.isnull()
test_ind  = y_test.isnull()

y_train.loc[train_ind] = 0
y_test.loc[test_ind] = 0

for model in tqdm(models):
    #y_train.loc[train_ind] += pd.Series(model.predict(X_train_test)).values
    y_test.loc[test_ind]   += pd.Series(model.predict(X_test_test)).values
    

#y_train.loc[train_ind] /= len(models)
y_test.loc[test_ind]   /= len(models)

y_train = y_train.to_frame()
y_test = y_test.to_frame()
y_train.columns = [label_name.replace('f001_', '')]
y_test.columns = [label_name.replace('f001_', '')]

y_train.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')
y_test.add_prefix(PREF+'_').to_pickle(f'../data/test_{PREF}.pkl')


#==============================================================================
utils.end(__file__)
#utils.stop_instance()



