#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:45:38 2018

@author: kazuki.onodera
"""


import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm

import utils, utils_metric
utils.start(__file__)
#==============================================================================


USE_FEATURES = 10
MOD_FEATURES = 90
MOD_N = 3

FILE_NAME = '1217-1'

# =============================================================================
# load
# =============================================================================
COL = pd.read_csv('LOG/imp_used_1217-1.csv').head(USE_FEATURES + (MOD_FEATURES*MOD_N) ).feature.tolist()

PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

files_te = [f'../feature/test_{c}.pkl' for c in COL]
sw = False
for i in files_te:
    if os.path.exists(i)==False:
        print(i)
        sw = True
if sw:
    raise Exception()

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
print(f'X.shape {X.shape}')

gc.collect()

feature_set = {}
for j in range(MOD_N):
    col = COL[:USE_FEATURES]
    col += [c for i,c in enumerate(COL[USE_FEATURES:]) if i%MOD_N==j]
    feature_set[j] = col


X_test = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_te, mininterval=10)
               ], axis=1)[COL]

# =============================================================================
# save
# =============================================================================

for i in range(MOD_N):
    utils.to_pkl_gzip(X[feature_set[i]], f'../data/X_train_{i}_{FILE_NAME}.pkl')
    utils.to_pkl_gzip(X_test[feature_set[i]], f'../data/X_test_{i}_{FILE_NAME}.pkl')
    
    
    