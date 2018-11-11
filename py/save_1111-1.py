#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 00:30:39 2018

@author: Kazuki

save train, test

"""

import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm

import utils
utils.start(__file__)
#==============================================================================

TRAIN_FILE_PATH = '../data/tr_1111-1.pkl'
TEST_FILE_PATH  = '../data/te_1111-1.pkl'

N_FEATURES = 350

# =============================================================================
# load
# =============================================================================
COL = pd.read_csv('LOG/imp_801_cv.py.csv').head(N_FEATURES).feature.tolist()

PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)[COL]

utils.to_pkl_gzip(X, TRAIN_FILE_PATH)

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

del X; gc.collect()


# =============================================================================
# test
# =============================================================================


files_te = []
for pref in PREFS:
    files_te += glob(f'../data/test_{pref}*.pkl')


def read(f):
    df = pd.read_pickle(f)
    col = list( set(df.columns) & set(COL) )
    return df[col]

X_test = pd.concat([
                read(f) for f in tqdm(files_te, mininterval=60)
               ], axis=1)[COL]


utils.to_pkl_gzip(X_test, TEST_FILE_PATH)

#==============================================================================
utils.end(__file__)
utils.stop_instance()
