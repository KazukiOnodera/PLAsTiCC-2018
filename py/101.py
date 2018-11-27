#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:43:32 2018

@author: Kazuki





In [17]: tr_log
Out[17]: 
         object_id   date  passband         flux  flux_norm1  flux_norm2
0              615  59748         1  -816.434326   -1.235849    0.429904
1              615  59748         2  -544.810303   -0.824688    0.841065
2              615  59748         3  -471.385529   -0.713543    0.952209
"""

import numpy as np
import pandas as pd
import os
import utils

def ddf_to_wfd(df, n):
    
    return df


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    # =============================================================================
    # train
    # =============================================================================
    tr = pd.read_pickle('../data/train.pkl')
    tr_log = pd.read_pickle('../data/train_log.pkl')
    
    
    
    tr_ddf = 
    tr_log = ddf_to_wfd(tr_log, 2)
    
    tr_log.to_pickle('../data/train_log2.pkl')
    
    
    utils.end(__file__)

