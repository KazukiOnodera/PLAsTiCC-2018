#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:31:28 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils


COMMENT = 'normalize'

EXE_SUBMIT = True

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    SUBMIT_FILE_PATH_out = '../output/1015-2_norm.csv.gz'
    SUBMIT_FILE_PATH_in = '../output/1015-2.csv.gz'
    
    sub = pd.read_csv(SUBMIT_FILE_PATH_in)
    
    sub.iloc[:, 1:] /= sub.iloc[:, 1:].sum(1)
    
    sub.to_csv(SUBMIT_FILE_PATH_out, index=False, 
               compression='gzip')
    
    # =============================================================================
    # submission
    # =============================================================================
    if EXE_SUBMIT:
        print('submit')
        utils.submit(SUBMIT_FILE_PATH_out, COMMENT)
    
    utils.end(__file__)


