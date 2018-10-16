#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:25:19 2018

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
from multiprocessing import cpu_count, Pool

import utils
utils.start(__file__)
#==============================================================================

SUBMIT_FILE_PATH = '../output/1016-1.csv.gz'

COMMENT = '1015-2 + bin'

EXE_SUBMIT = True

DROP = ['f001_hostgal_specz']

SEED = np.random.randint(9999)
print('SEED:', SEED)

NFOLD = 7

LOOP = 1

param = {
         'objective': 'binary',
         'metric': 'binary_logloss',
         
         'learning_rate': 0.01,
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.5,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }

# =============================================================================
# load
# =============================================================================

files_tr = sorted(glob('../data/train_f*.f'))

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
y = utils.load_target().target

X.drop(DROP, axis=1, inplace=True)

#target_dict = {}
#target_dict_r = {}
#for i,e in enumerate(y.sort_values().unique()):
#    target_dict[e] = i
#    target_dict_r[i] = e
#
#y = y.replace(target_dict)

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

COL = X.columns.tolist()

#CAT = list( set(X.columns)&set(utils_cat.ALL))
#print(f'CAT: {CAT}')

# =============================================================================
# cv
# =============================================================================


y_unique = y.unique()
model_di = {}

for i in y_unique:
    print('build', i)
    
    dtrain = lgb.Dataset(X, (y==i)*1, #categorical_feature=CAT, 
                         free_raw_data=False)
    gc.collect()

    model_all = []
    for j in range(LOOP):
        gc.collect()
        param['seed'] = np.random.randint(9999)
        ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD, 
                             early_stopping_rounds=100, verbose_eval=50,
                             seed=SEED)
        model_all += models
    
    model_di[i] = model_all

#result = f"CV auc-mean: {ret['multi_logloss-mean'][-1]} + {ret['multi_logloss-stdv'][-1]}"
#print(result)
#
#utils.send_line(result)
#imp = ex.getImp(model_all)
#imp['split'] /= imp['split'].max()
#imp['gain'] /= imp['gain'].max()
#imp['total'] = imp['split'] + imp['gain']
#
#imp.sort_values('total', ascending=False, inplace=True)
#imp.reset_index(drop=True, inplace=True)
#
#
#imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

# =============================================================================
# test
# =============================================================================

files_te = sorted(glob('../data/test_f*.f'))

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files_te, mininterval=60)
               ], axis=1)[COL]


sub = pd.read_csv('../input/sample_submission.csv.zip')[['object_id']]

#def multi(args):
#    i, model = args
#    y_pred = model.predict(X_test)
#    np.savez(f'../data/{i}.npz', x=y_pred)
#    return
#
#for i in y_unique:
#    print('predict', i)
#    
#    model_all = model_di[i]
#    argss = list(zip(range(len(model_all)), model_all))
#    pool = Pool(5)
#    pool.map(multi, argss)
#    pool.close()
#    
#    files = glob('../data/*.npz')
#    for j,file in enumerate(tqdm(files)):
#        y_pred_ = np.load(file)['x']
#        if j==0:
#            y_pred = y_pred_
#        else:
#            y_pred += y_pred_
#    
#    y_pred /= len(model_all)
#    
#    sub[f'class_{i}'] = y_pred
#    os.system('rm -rf ../data/*.npz')


for i in y_unique:
    print('predict', i)
    
    model_all = model_di[i]
    for j,model in enumerate(tqdm(model_all)):
        y_pred_ = model.predict(X_test)
        if j==0:
            y_pred = y_pred_
        else:
            y_pred += y_pred_
    
    y_pred /= len(model_all)
    
    sub[f'class_{i}'] = y_pred

# Compute preds_99 as the proba of class not being any of the others
# preds_99 = 0.1 gives 1.769
preds_99 = np.ones(sub.shape[0])
for i in range(sub.shape[1]-1):
    preds_99 *= (1 - sub.iloc[:, i+1])
sub['class_99'] = preds_99


sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

png = f'LOG/sub_{__file__}.png'
utils.savefig_sub(sub, png)
utils.send_line('DONE!', png)

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)





#==============================================================================
utils.end(__file__)
utils.stop_instance()



