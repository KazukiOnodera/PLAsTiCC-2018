#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:36:17 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
from glob import glob
import utils

files = ['../output/1022-1_Giba-post2-1103.csv.gz', 
         '../output/1109-3_post.csv.gz', 
         '../output/1109-1_post.csv.gz',
         '../output/1109-3.csv.gz', ]

def plt(f):
    sub = pd.read_csv(f)
    print(f)
    print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))

for f in files:
    plt(f)

