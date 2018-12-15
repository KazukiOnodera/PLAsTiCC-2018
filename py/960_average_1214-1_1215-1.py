#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:12:06 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils

EXE_SUBMIT = True

SUBMIT_FILE_PATH = '../output/1214-1_1215-1.csv.gz'

COMMENT = 'mlogloss(0.921) + wmlogloss(???)'

sub1 = pd.read_csv('../output/1214-1.csv.gz')
sub2 = pd.read_csv('../output/1215-1.csv.gz')


sub = (sub1 + sub2)/2
sub.iloc[:, 1:] = sub.iloc[:, 1:].values / sub.iloc[:, 1:].sum(1).values[:,None]

print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))


sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')


# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)


