#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:01:08 2018

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

import utils
utils.start(__file__)
#==============================================================================

SUBMIT_FILE_PATH = '../output/1022-1.csv.gz'

COMMENT = 'f006~010'

EXE_SUBMIT = True

DROP = ['f001_hostgal_specz']

SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5

LOOP = 5

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
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
[print(f) for f in files_tr]

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files_tr, mininterval=60)
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

COL = X.columns.tolist()
#CAT = list( set(X.columns)&set(utils_cat.ALL))
#print(f'CAT: {CAT}')

# =============================================================================
# cv
# =============================================================================

dtrain = lgb.Dataset(X, y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
for i in range(LOOP):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         feval=utils.lgb_multi_weighted_logloss,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/LOOP) * 1.3)

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)

imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

png = f'LOG/imp_{__file__}.png'
utils.savefig_imp(imp, png, x='total', title=f'{__file__}')
utils.send_line(result, png)


COL = imp[imp.gain>0].feature.tolist()

# =============================================================================
# model
# =============================================================================

dtrain = lgb.Dataset(X[COL], y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()


np.random.seed(SEED)

model_all = []
for i in range(LOOP):
    print('building', i)
    gc.collect()
    param['seed'] = np.random.randint(9999)
    model = lgb.train(param, dtrain, num_boost_round=nround_mean, valid_sets=None, 
                      valid_names=None, fobj=None, feval=None, init_model=None, 
                      feature_name='auto', categorical_feature='auto', 
                      early_stopping_rounds=None, evals_result=None, 
                      verbose_eval=True, learning_rates=None, 
                      keep_training_booster=False, callbacks=None)
    
    model_all.append(model)


del dtrain, X; gc.collect()


# =============================================================================
# test
# =============================================================================

files_te = sorted(glob('../data/test_f*.f'))

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files_te, mininterval=60)
               ], axis=1)[COL]

for i,model in enumerate(tqdm(model_all)):
    y_pred = model.predict(X_test)
    if i==0:
        y_pred_all = y_pred
    else:
        y_pred_all += y_pred

y_pred_all /= len(model_all)

sub = pd.read_csv('../input/sample_submission.csv.zip')
df = pd.DataFrame(y_pred_all, columns=sub.columns[1:-1])

# Compute preds_99 as the proba of class not being any of the others
# preds_99 = 0.1 gives 1.769
preds_99 = np.ones(df.shape[0])
for i in range(df.shape[1]):
    preds_99 *= (1 - df.iloc[:, i])
df['class_99'] = preds_99


sub = pd.concat([sub[['object_id']], df], axis=1)

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

sub.iloc[:, 1:].hist(bins=30, figsize=(16, 12))

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
#utils.stop_instance()


