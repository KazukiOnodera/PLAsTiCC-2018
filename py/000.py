#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:12:10 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils

os.system(f'rm -rf ../data')
os.system(f'mkdir ../data')
os.system(f'rm -rf ../feature')
os.system(f'mkdir ../feature')

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    # train
    train     = pd.read_csv('../input/training_set_metadata.csv')
    train.to_feather('../data/train.f')
    train[['target']].to_feather('../data/target.f')
    
    train_log = pd.read_csv('../input/training_set.csv.zip')
    train_log.to_feather('../data/train_log.f')
    
    # test
    test     = pd.read_csv('../input/test_set_metadata.csv.zip')
    test.to_feather('../data/test.f')
    test_log = pd.read_csv('../input/test_set.csv.zip')
    test_log.to_feather('../data/test_log.f')
    
    
    utils.end(__file__)

