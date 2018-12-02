#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:40:40 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils

PREF = 'f001'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

#def mk_feats(df):
#    df['hostgal_specz-m-hostgal_photoz'] = df['hostgal_specz'] - df['hostgal_photoz']
#    df['hostgal_specz-d-hostgal_photoz'] = df['hostgal_specz'] / df['hostgal_photoz']
#    df['hostgal_photoz-d-hostgal_photoz_err'] = df['hostgal_photoz'] / df['hostgal_photoz_err']
#    df['hostgal_specz-d-hostgal_photoz_err'] = df['hostgal_specz'] / df['hostgal_photoz_err']
#    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    train = utils.load_train().drop(['object_id', 'target'], axis=1)
    train.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')
    train_aug = pd.read_pickle('../data/train_aug.pkl').drop(['object_id', 'object_id_bk', 'target'], axis=1)
    train_aug.add_prefix(PREF+'_').to_pickle(f'../data/train_aug_{PREF}.pkl')
    
    
    test  = utils.load_test().drop(['object_id'], axis=1)
    test.loc[test.hostgal_photoz==0, 'hostgal_specz'] = 0
    test = test.add_prefix(PREF+'_')
    test.to_pickle(f'../data/test_{PREF}.pkl')
    utils.save_test_features(test)
    
    utils.end(__file__)

