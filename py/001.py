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

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    train = utils.load_train().drop(['object_id', 'target'], axis=1)
    train.add_perfix(PREF+'_').to_feather(f'../data/train_{PREF}.f')
    
    test  = utils.load_test().drop(['object_id'], axis=1)
    test.add_perfix(PREF+'_').to_feather(f'../data/test_{PREF}.f')
    
    utils.end(__file__)

