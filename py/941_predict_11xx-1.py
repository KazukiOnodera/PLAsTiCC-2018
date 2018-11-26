#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:20:37 2018

@author: kazuki.onodera


separate gal & exgal

oversampling




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

SUBMIT_FILE_PATH = '../output/11??-1.csv.gz'

COMMENT = 'gal: 90 exgal: 112 / oversampling'

EXE_SUBMIT = True


SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5

LOOP = 3

param = {
         'objective': 'multiclass',
#         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.5,
         'max_depth': 3,
         'num_leaves': 63,
         'max_bin': 255,
         
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


#USE_FEATURES = 100



# =============================================================================
# load
# =============================================================================
#COL_gal   = pd.read_csv('LOG/imp_802_cv_separate.py_gal.csv').head(USE_FEATURES ).feature.tolist()
#COL_exgal = pd.read_csv('LOG/imp_802_cv_separate.py_exgal.csv').head(USE_FEATURES ).feature.tolist()

COL_gal   = pd.read_pickle('../data/807_gal.pkl').columns.tolist()
COL_exgal = pd.read_pickle('../data/807_exgal.pkl').columns.tolist()

COL = list(set(COL_gal + COL_exgal))

PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')


files_te = [f'../feature/test_{c}.pkl' for c in COL]
files_te = sorted(files_te)
sw = False
for i in files_te:
    if os.path.exists(i)==False:
        print(i)
        sw = True
if sw:
    raise Exception()
else:
    print('all test file exist!')


X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)[COL]
y = utils.load_target().target




target_dict = {}
target_dict_r = {}
for i,e in enumerate(y.sort_values().unique()):
    target_dict[e] = i
    target_dict_r[i] = e

y = y.replace(target_dict)

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')

gc.collect()

# =============================================================================
# separate
# =============================================================================

is_gal = pd.read_pickle('../data/tr_is_gal.pkl')

X_gal = X[is_gal][COL_gal]
y_gal = y[is_gal]

X_exgal = X[~is_gal][COL_exgal]
y_exgal = y[~is_gal]


def target_replace(y):
    target_dict = {}
    target_dict_r = {}
    for i,e in enumerate(y.sort_values().unique()):
        target_dict[e] = i
        target_dict_r[i] = e
    
    return y.replace(target_dict), target_dict_r


y_gal, di_gal = target_replace(y_gal)
y_exgal, di_exgal = target_replace(y_exgal)


del X, y
gc.collect()

print(f'X_gal.shape: {X_gal.shape}')
print(f'X_exgal.shape: {X_exgal.shape}')

# =============================================================================
# oversampling
# =============================================================================
"""
train:
    is_gal	False	True
    ddf		
    0	3947	1785
    1	1576	540

test:
    is_gal	False	True
    ddf		
    0	3069681	390283
    1	32699	227

(390283 / 227) / (1785 / 540) == 520
(3069681 / 32699) / (3947 / 1576) == 37


"""

print('oversampling')

from sklearn.model_selection import GroupKFold


is_ddf = pd.read_pickle('../data/train.pkl').ddf==1

# ======== for gal ========
X_gal['g'] = np.arange(X_gal.shape[0]) % NFOLD
X_gal_d0 = X_gal[~is_ddf]
X_gal_d1 = X_gal[is_ddf]
li = [X_gal_d0.copy() for i in range(520)]

X_gal = pd.concat([X_gal_d1]+li, ignore_index=True)

group_gal = X_gal.g

y_gal_d0 = y_gal[~is_ddf]
y_gal_d1 = y_gal[is_ddf]
li = [y_gal_d0.copy() for i in range(520)]
y_gal = pd.concat([y_gal_d1]+li, ignore_index=True)

del li, X_gal_d0, X_gal_d1, X_gal['g'], y_gal_d0, y_gal_d1


# ======== for exgal ========
X_exgal['g'] = np.arange(X_exgal.shape[0]) % NFOLD
X_exgal_d0 = X_exgal[~is_ddf]
X_exgal_d1 = X_exgal[is_ddf]
li = [X_exgal_d0.copy() for i in range(37)]

X_exgal = pd.concat([X_exgal_d1]+li, ignore_index=True)

group_exgal = X_exgal.g

y_exgal_d0 = y_exgal[~is_ddf]
y_exgal_d1 = y_exgal[is_ddf]
li = [y_exgal_d0.copy() for i in range(37)]
y_exgal = pd.concat([y_exgal_d1]+li, ignore_index=True)

del li, X_exgal_d0, X_exgal_d1, X_exgal['g'], y_exgal_d0, y_exgal_d1

group_kfold = GroupKFold(n_splits=NFOLD)

print(f'X_gal.shape: {X_gal.shape}')
print(f'X_exgal.shape: {X_exgal.shape}')

gc.collect()


# =============================================================================
# cv(gal)
# =============================================================================
print('==== GAL ====')
param['num_class'] = 5

dtrain = lgb.Dataset(X_gal, y_gal.values, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

#model_all = []
nround_mean = 0
wloss_list = []
oofs_gal = []
for i in range(2):
    
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         fobj=utils_metric.wloss_objective_gal, 
                         feval=utils_metric.wloss_metric_gal,
                         early_stopping_rounds=100, verbose_eval=50,
                         folds=group_kfold.split(X_gal, y_gal, group_gal),
                         seed=SEED)
    oof = ex.eval_oob(X_gal, y_gal.values, models, SEED, stratified=True, shuffle=True, 
                         n_class=True)
    oofs_gal.append(oof)
#    model_all += models
    nround_mean += len(ret['wloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/2) * 1.3)
utils.send_line(f'nround_mean: {nround_mean}')

result = f"CV GAL wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
utils.send_line(result)

# =============================================================================
# model(gal)
# =============================================================================

gc.collect()


np.random.seed(SEED)

model_all_gal = []
for i in range(LOOP):
    
    gc.collect()
    
    print(f'LOOP:{i}')
    gc.collect()
    param['seed'] = np.random.randint(9999)
    model = lgb.train(param, dtrain, num_boost_round=nround_mean, 
                      fobj=utils_metric.wloss_objective_gal, 
                      feval=utils_metric.wloss_metric_gal,
                      valid_names=None, init_model=None, 
                      feature_name='auto', categorical_feature='auto', 
                      early_stopping_rounds=None, evals_result=None, 
                      verbose_eval=True, learning_rates=None, 
                      keep_training_booster=False, callbacks=None)

    model_all_gal.append(model)

del X_gal, y_gal

# =============================================================================
# cv(exgal)
# =============================================================================
print('==== EXGAL ====')
param['num_class'] = 9

dtrain = lgb.Dataset(X_exgal, y_exgal.values, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

#model_all = []
nround_mean = 0
wloss_list = []
oofs_exgal = []
for i in range(2):
    
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         fobj=utils_metric.wloss_objective_exgal, 
                         feval=utils_metric.wloss_metric_exgal,
                         early_stopping_rounds=100, verbose_eval=50,
                         folds=group_kfold.split(X_exgal, y_exgal, group_exgal),
                         seed=SEED)
    oof = ex.eval_oob(X_exgal, y_exgal.values, models, SEED, stratified=True, shuffle=True, 
                         n_class=True)
    oofs_exgal.append(oof)
#    model_all += models
    nround_mean += len(ret['wloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/2) * 1.3)
utils.send_line(f'nround_mean: {nround_mean}')

result = f"CV EXGAL wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)
utils.send_line(result)

# =============================================================================
# model(exgal)
# =============================================================================

gc.collect()


np.random.seed(SEED)

model_all_exgal = []
for i in range(LOOP):
    
    gc.collect()
    
    print(f'LOOP:{i}')
    gc.collect()
    param['seed'] = np.random.randint(9999)
    model = lgb.train(param, dtrain, num_boost_round=nround_mean, 
                      fobj=utils_metric.wloss_objective_exgal, 
                      feval=utils_metric.wloss_metric_exgal,
                      valid_names=None, init_model=None, 
                      feature_name='auto', categorical_feature='auto', 
                      early_stopping_rounds=None, evals_result=None, 
                      verbose_eval=True, learning_rates=None, 
                      keep_training_booster=False, callbacks=None)

    model_all_exgal.append(model)






# =============================================================================
# test TODO: edit
# =============================================================================

X_test = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_te, mininterval=10)
               ], axis=1)

gc.collect()

is_gal = pd.read_pickle('../data/te_is_gal.pkl')

X_test_gal   = X_test[is_gal][COL_gal]
X_test_exgal = X_test[~is_gal][COL_exgal]

del X_test; gc.collect()

for i,(model_gal,model_exgal) in enumerate(zip(model_all_gal, model_all_exgal)):
    y_pred_gal = model_gal.predict(X_test_gal)
    y_pred_gal = utils_metric.softmax(y_pred_gal)
    y_pred_exgal = model_exgal.predict(X_test_exgal)
    y_pred_exgal = utils_metric.softmax(y_pred_exgal)
    if i==0:
        y_pred_all_gal   = y_pred_gal
        y_pred_all_exgal = y_pred_exgal
    else:
        y_pred_all_gal   += y_pred_gal
        y_pred_all_exgal += y_pred_exgal

y_pred_all_gal /= int(LOOP)
y_pred_all_exgal /= int(LOOP)


sub_gal   = pd.read_pickle('../data/te_oid_gal.pkl')
sub_exgal = pd.read_pickle('../data/te_oid_exgal.pkl')

sub_gal = pd.concat([sub_gal, pd.DataFrame(y_pred_all_gal)], axis=1)
sub_gal.columns = ['object_id'] + [f'class_{i}' for i in [6, 16, 53, 65, 92]]

sub_exgal = pd.concat([sub_exgal, pd.DataFrame(y_pred_all_exgal)], axis=1)
sub_exgal.columns = ['object_id'] + [f'class_{i}' for i in [15, 42, 52, 62, 64, 67, 88, 90, 95]]


sub = pd.concat([sub_gal, sub_exgal], ignore_index=True).fillna(0)
col = ['object_id', 'class_6', 'class_15', 'class_16', 'class_42', 'class_52',
       'class_53', 'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 
       'class_90', 'class_92', 'class_95']

sub = sub[col]


# class_99
sub.to_pickle(f'../data/y_pred_raw_{__file__}.pkl')
utils.postprocess(sub, method='oli')

print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))


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


