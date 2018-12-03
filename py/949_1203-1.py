#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 03:20:48 2018

@author: Kazuki
"""

import pandas as pd
import utils


SUBMIT_FILE_PATH = '../output/1203-1.csv.gz'
COMMENT = '1120-1 + taguchi-959'


sub1 = pd.read_csv('../output/1120-1.csv.gz')

sub2 = pd.read_csv('../FROM_MYTEAM/LB0.959_Booster_weight-multi-logloss-0.577933_2018-11-29-20-26-36_setzero.csv.gz')

sub = sub1 + sub2
sub /= 2

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')
utils.submit(SUBMIT_FILE_PATH, COMMENT)

