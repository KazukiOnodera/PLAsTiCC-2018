#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:31:28 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils

SUBMIT_FILE_PATH_in = '../output/1015-2.csv.gz'

SUBMIT_FILE_PATH_out = '../output/1015-2_norm.csv.gz'

COMMENT = 'normalize'

EXE_SUBMIT = True

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    sub = pd.read_csv(SUBMIT_FILE_PATH_in)
    
    tmp = sub.iloc[:, 1:].sum(1)
    for i in range(1, sub.shape[1]):
        sub.iloc[:, i] /= tmp
    
    sub.to_csv(SUBMIT_FILE_PATH_out, index=False, 
               compression='gzip')
    
    # =============================================================================
    # submission
    # =============================================================================
    if EXE_SUBMIT:
        print('submit')
        utils.submit(SUBMIT_FILE_PATH_out, COMMENT)
    
    utils.end(__file__)


