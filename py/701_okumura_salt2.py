#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:36:39 2018

@author: kazuki.onodera

奥村神から授かった特徴

"""

import numpy as np
import pandas as pd
import os
import utils

PREF = 'f701'

file_tr = '../data/train_sncosmo_salt2_fitting_features_20181121.csv'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    df = pd.read_csv(file_tr)
    df.sort_values('object_id', inplace=True)
    del df['object_id']
    df = df.add_prefix(f'{PREF}_')
    df.to_pickle(f'../data/train_{PREF}.pkl')
    
    utils.end(__file__)


