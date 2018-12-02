#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 18:28:54 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

import os

import utils

NFEATURES = 150
NTHREADS = 2



COL = pd.read_csv('LOG/imp_702_52vs90-1.py-2.csv').head(NFEATURES).feature.tolist()
#COL = pd.read_csv('LOG/imp_801_cv.py-2.csv').head(NFEATURES).feature.tolist()

#COL_gal   = pd.read_csv('LOG/imp_802_cv_separate.py_gal.csv').head(NFEATURES ).feature.tolist()
#COL_exgal = pd.read_csv('LOG/imp_802_cv_separate.py_exgal.csv').head(NFEATURES ).feature.tolist()
#COL = list(set(COL_gal + COL_exgal))


PREFS = sorted(set([c.split('_')[0] for c in COL]))


files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

files_te = [f'../feature/test_{c}.pkl' for c in COL]

gen_features = []
for i in files_te:
    if os.path.exists(i)==False:
        gen_features.append(i[16:-4])
gen_features = sorted(gen_features)
print(gen_features)

gen_prefs = sorted(set([x.split('_')[0] for x in gen_features]))

def multi(pref):
    df = pd.read_pickle(f'../data/test_{pref}.pkl.gz')
    col = list( set(df.columns) & set(gen_features) )
    print(pref, col)
    utils.save_test_features(df[col])
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    pool = Pool(NTHREADS)
    pool.map(multi, gen_prefs)
    pool.close()
    
    
    utils.end(__file__)


