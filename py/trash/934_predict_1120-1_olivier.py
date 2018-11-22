#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:38:49 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils


SUBMIT_FILE_PATH = '../output/1120-1_olivier.csv.gz'

COMMENT = '1120-1 + olivier'

classes_gal = [6, 16, 53, 65, 92]
classes_exgal = [15, 42, 52, 62, 64, 67, 88, 90, 95]

# =============================================================================
# main
# =============================================================================

sub = pd.read_pickle('../data/y_pred_raw_934_predict_1120-1.py.pkl')

oid_gal   = pd.read_pickle('../data/te_oid_gal.pkl').object_id
oid_exgal = pd.read_pickle('../data/te_oid_exgal.pkl').object_id
sub.loc[sub.object_id.isin(oid_gal),  [f'class_{i}' for i in classes_exgal]] = 0
sub.loc[sub.object_id.isin(oid_exgal),[f'class_{i}' for i in classes_gal]] = 0

preds_99 = np.ones(sub.shape[0])
for i in range(1, sub.shape[1]):
    preds_99 *= (1 - sub.iloc[:, i])

sub['class_99'] = 0.14 * preds_99 / np.mean(preds_99)
val = sub.iloc[:, 1:].values
val = np.clip(a=val, a_min=1e-15, a_max=1 - 1e-15)
val /= val.sum(1)[:,None]
sub.iloc[:, 1:] = val


# check
sub1 = pd.read_csv('../output/1120-1.csv.gz')

print( sub1.class_99.corr(sub.class_99) )
print( sub1.class_99.corr(sub.class_99, 'spearman') )





sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
print('submit')
utils.submit(SUBMIT_FILE_PATH, COMMENT)



