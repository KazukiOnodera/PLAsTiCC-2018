#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 08:05:54 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm
from scipy.stats import kurtosis
from multiprocessing import cpu_count, Pool
from tsfresh.feature_extraction import extract_features

import sys
argvs = sys.argv

import utils

PREF = 'f021'

max_index = 30



# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    imp = pd.read_csv(utils.IMP_FILE).head(utils.GENERATE_FEATURE_SIZE)
    usecols = imp[imp.feature.str.startswith(f'{PREF}')][imp.gain>0].feature.tolist()
    usecols = [c.replace(f'{PREF}_', '') for c in usecols]
    usecols += ['object_id']
    
    df = pd.concat([pd.read_pickle(f) for f in tqdm(glob(f'../data/tmp_{PREF}*'))], 
                    ignore_index=True)
    df.sort_values(f'{PREF}_object_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    del df[f'{PREF}_object_id']
    utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
    utils.save_test_features(df[usecols])
    os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)

