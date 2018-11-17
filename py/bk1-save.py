#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 05:18:25 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import gc
from glob import glob
from tqdm import tqdm
import utils


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    files = sorted(glob('../data/test_fbk1*'))
    for file in tqdm(files):
        df = pd.read_pickle(file)
        utils.save_test_features(df)
        del df; gc.collect()
    
    utils.end(__file__)

