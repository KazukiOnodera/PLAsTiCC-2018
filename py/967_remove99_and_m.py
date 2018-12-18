#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 07:25:35 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils



SUBMIT_FILE_PATH = '../output/1216_without99_only-wm.csv.gz'


sub1m = pd.read_csv('../output/1216-m-1.csv.gz').set_index('object_id')
sub2m = pd.read_csv('../output/1216-m-2.csv.gz').set_index('object_id')
sub3m = pd.read_csv('../output/1216-m-3.csv.gz').set_index('object_id')

sub1wm = pd.read_csv('../output/1216-wm-1.csv.gz').set_index('object_id')
sub2wm = pd.read_csv('../output/1216-wm-2.csv.gz').set_index('object_id')
sub3wm = pd.read_csv('../output/1216-wm-3.csv.gz').set_index('object_id')

#norm
def norm(df):
    del df['class_99']
    df.iloc[:, 1:] = df.iloc[:, 1:].values / df.iloc[:, 1:].sum(1).values[:,None]
    print('df.mean:', df.iloc[:, 1:].sum(1).mean())
    return

norm(sub1m)
norm(sub2m)
norm(sub3m)
norm(sub1wm)
norm(sub2wm)
norm(sub3wm)

sub = (sub1wm + sub2wm + sub3wm)/3

print('sub.mean:', sub.iloc[:, 1:].sum(1).mean())
print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))
print(sub.iloc[:, 1:].sum())


sub.to_csv(SUBMIT_FILE_PATH, compression='gzip')



