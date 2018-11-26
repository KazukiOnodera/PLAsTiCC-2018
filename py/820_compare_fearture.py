#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:40:33 2018

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
#         'num_class': 14,
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

USE_FEATURES = 300

# =============================================================================
# load
# =============================================================================
COL_gal   = pd.read_csv('LOG/imp_802_cv_separate.py_gal.csv').head(USE_FEATURES ).feature.tolist()
COL_exgal = pd.read_csv('LOG/imp_802_cv_separate.py_exgal.csv').head(USE_FEATURES ).feature.tolist()

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


X_tr = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)[COL]
y = utils.load_target().target


X_te = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_te, mininterval=10)
               ], axis=1)[COL]

gc.collect()



target_dict = {}
target_dict_r = {}
for i,e in enumerate(y.sort_values().unique()):
    target_dict[e] = i
    target_dict_r[i] = e

y = y.replace(target_dict)

if X_tr.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_tr.columns[X_tr.columns.duplicated()] }')
print('no dup :) ')
print(f'X_tr.shape {X_tr.shape}')

gc.collect()

# =============================================================================
# separate
# =============================================================================

tr_is_gal = pd.read_pickle('../data/tr_is_gal.pkl')
te_is_gal = pd.read_pickle('../data/te_is_gal.pkl')

X_tr_gal = X_tr[tr_is_gal][COL_gal]
X_te_gal = X_te[te_is_gal][COL_gal]

X_tr_exgal = X_tr[~tr_is_gal][COL_exgal]
X_te_exgal = X_te[~te_is_gal][COL_exgal]

del X_tr, X_te; gc.collect()


# =============================================================================
# compare
# =============================================================================

for c in X_tr_gal.columns:
    print(f'{c}: train:{X_tr_gal[c].std():.3f}     test:{X_te_gal[c].std():.3f}')


for c in X_tr_exgal.columns:
    print(f'{c}: train:{X_tr_exgal[c].std():.3f}     test:{X_te_exgal[c].std():.3f}')





