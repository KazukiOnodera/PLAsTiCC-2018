#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:13:29 2018

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

SUBMIT_FILE_PATH = '../output/1015-4.csv.gz'

COMMENT = '1015-2 + weight'

EXE_SUBMIT = True

DROP = ['f001_hostgal_specz']

SEED = np.random.randint(9999)
print('SEED:', SEED)

NFOLD = 4

LOOP = 2

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.1,
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

classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]

class_weight = {6: 1, 
                15: 2, 
                16: 1, 
                42: 1, 
                52: 1, 
                53: 1, 
                62: 1, 
                64: 2, 
                65: 1, 
                67: 1, 
                88: 1, 
                90: 1, 
                92: 1, 
                95: 1}


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
def lgb_multi_weighted_logloss(y_preds, train_data):
    """
    @author olivier https://www.kaggle.com/ogrellier
    https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data/code
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    y_true = train_data.get_label()
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

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
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


dtrain = lgb.Dataset(X, y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
for i in range(LOOP):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 30, nfold=NFOLD, 
                         feval=lgb_multi_weighted_logloss,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models

result = f"CV auc-mean: {ret['multi_logloss-mean'][-1]} + {ret['multi_logloss-stdv'][-1]}"
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

# =============================================================================
# test
# =============================================================================

files_te = sorted(glob('../data/test_f*.f'))

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files_te, mininterval=60)
               ], axis=1)[COL]

X_test.drop(DROP, axis=1, inplace=True)

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
utils.send_line(result, png)

raise
# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)





#==============================================================================
utils.end(__file__)
utils.stop_instance()


