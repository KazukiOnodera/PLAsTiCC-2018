#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:18:01 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd

# kernel best(1.135)
sub1 = pd.read_csv('/Users/kazuki.onodera/Downloads/single_predictions.csv').set_index('object_id')

sub2 = pd.read_csv('/Users/kazuki.onodera/Downloads/1022-1_Giba-post2-1103.csv.gz').set_index('object_id')
sub3 = pd.read_csv('/Users/kazuki.onodera/Downloads/1109-3_post.csv.gz').set_index('object_id')


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

sub.to_csv('1.113+1.160+kernel1.135.csv.gz', compression='gzip')


"""
sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True)

class_99    0.578037
class_15    0.074243
class_64    0.063760
class_52    0.056890
class_92    0.054826
class_16    0.029227
class_42    0.027932
class_65    0.027054
class_88    0.025142
class_62    0.023343
class_67    0.023306
class_95    0.015094
class_90    0.000731
class_53    0.000415
dtype: float64


sub.iloc[:, 1:].mean()

class_15    0.082748
class_16    0.028134
class_42    0.099229
class_52    0.096878
class_53    0.001210
class_62    0.068725
class_64    0.054994
class_65    0.024477
class_67    0.059380
class_88    0.026652
class_90    0.126347
class_92    0.050831
class_95    0.023255
class_99    0.254093
dtype: float64



"""
