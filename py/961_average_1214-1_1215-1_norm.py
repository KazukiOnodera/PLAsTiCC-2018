#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 06:20:38 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils

EXE_SUBMIT = True

SUBMIT_FILE_PATH = '../output/1214-1_1215-1_norm.csv.gz'

COMMENT = 'mlogloss(0.921) + wmlogloss(???) norm'

sub1 = pd.read_csv('../output/1214-1.csv.gz')
sub2 = pd.read_csv('../output/1215-1.csv.gz')

#norm
sub1.iloc[:, 1:] = sub1.iloc[:, 1:].values / sub1.iloc[:, 1:].sum(1).values[:,None]
sub2.iloc[:, 1:] = sub2.iloc[:, 1:].values / sub2.iloc[:, 1:].sum(1).values[:,None]

print('sub1.mean:', sub1.iloc[:, 1:].sum(1).mean())
print('sub2.mean:', sub2.iloc[:, 1:].sum(1).mean())


sub = (sub1 + sub2)/2
sub.iloc[:, 1:] = sub.iloc[:, 1:].values / sub.iloc[:, 1:].sum(1).values[:,None]

print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))
print('sub.mean:', sub.iloc[:, 1:].sum(1).mean())


sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')


# =============================================================================
# submission
# =============================================================================
#if EXE_SUBMIT:
#    print('submit')
#    utils.submit(SUBMIT_FILE_PATH, COMMENT)


