#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:05:17 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
#from multiprocessing import Pool
import utils

PREF = 'f006'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

usecols = ['object_id', 'passband']

def mk_feature(df, output_path):
    
    ct = pd.crosstab(df['object_id'], df['passband'])
    ct.columns = [f'passband{c}_cnt' for c in ct.columns]
    
    ct_norm = pd.crosstab(df['object_id'], df['passband'], normalize='index')
    ct_norm.columns = [f'passband{c}_norm' for c in ct_norm.columns]
    
    df = pd.concat([ct, ct_norm], axis=1)
    
    df.reset_index(drop=True, inplace=True)
    df.add_prefix(PREF+'_').to_feather(output_path)
    
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    mk_feature(pd.read_feather('../data/train_log.f')[usecols], f'../data/train_{PREF}.f')
    mk_feature(pd.read_feather('../data/test_log.f')[usecols],  f'../data/test_{PREF}.f')
    
    utils.end(__file__)


