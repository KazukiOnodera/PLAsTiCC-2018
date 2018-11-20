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

N_FEATURES = 300

COL = pd.read_csv('LOG/imp_801_cv.py-2.csv').head(N_FEATURES).feature.tolist()


PREFS = sorted(set([c.split('_')[0] for c in COL]))


files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

files_te = [f'../feature/test_{c}.pkl' for c in COL]

gen_features = []
for i in files_te:
    if os.path.exists(i)==False:
        gen_features.append(i[16:])


gen_prefs = sorted(set([x.split('_')[0] for x in gen_features]))

def multi(pref):
    df = pd.read_pickle(f'../data/test_{pref}.pkl.gz')
    col = list( set(df.columns) & set(gen_features) )
    utils.save_test_features(df[col])
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    pool = Pool(5)
    pool.map(multi, gen_prefs)
    pool.close()
    
    
    utils.end(__file__)


