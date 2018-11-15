#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:46:18 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd

# kernel best(1.135)
sub1 = pd.read_csv('/Users/kazuki.onodera/1.113+1.160+kernel1.135.csv.gz').set_index('object_id')

sub2 = pd.read_csv('/Users/kazuki.onodera/Downloads/Booster_weight-multi-logloss-0.612193_2018-11-11-04-49-01_res_lin_pure.csv').set_index('object_id')
sub3 = pd.read_csv('/Users/kazuki.onodera/Downloads/Booster_weight-multi-logloss-0.612193_2018-11-11-04-49-01.csv').set_index('object_id')


def norm(df):
    sum1 = df.sum(1)
    for c in df.columns:
        df[c] /= sum1
    return

norm(sub1)
norm(sub2)
norm(sub3)


sub = pd.DataFrame(index=sub1.index)

for c in sub1.columns:
    sub[c] = (sub1[c] + sub2[c] + sub3[c]) / 3

sub.to_csv('onodera1.021-and-guchioBest2.csv.gz', compression='gzip')

