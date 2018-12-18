#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:11:39 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils

EXE_SUBMIT = True

SEED = 71
np.random.seed(SEED)

FILE_in  = '../output/matsuken-875_onodera-884_taguchi-888u_akiyama-889u.csv.gz'
FILE_out = '../output/LB839_c99_uniform.csv.gz'

COMMENT = 'np.random.uniform(1.2, 1.4)'

sub = pd.read_csv(FILE_in)

sub.class_99 *= np.random.uniform(1.2, 1.4, size=sub.shape[0])


sub.iloc[:, 1:] = sub.iloc[:, 1:].values / sub.iloc[:, 1:].sum(1).values[:,None]

sub.to_csv(FILE_out, index=False, compression='gzip')


# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(FILE_out, COMMENT)



