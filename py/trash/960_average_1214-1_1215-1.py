#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:12:06 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils, utils_metric

EXE_SUBMIT = True

SUBMIT_FILE_PATH = '../output/1214-1_1215-1.csv.gz'

COMMENT = 'mlogloss(0.921) + wmlogloss(???)'

sub1 = pd.read_csv('../output/1214-1.csv.gz')
sub2 = pd.read_csv('../output/1215-1.csv.gz')

print(sub1.iloc[:,1:].sum(1).mean(), sub2.iloc[:,1:].sum(1).mean())
sub1.iloc[:, 1:] = sub1.iloc[:, 1:].values / sub1.iloc[:, 1:].sum(1).values[:,None]
sub2.iloc[:, 1:] = sub2.iloc[:, 1:].values / sub2.iloc[:, 1:].sum(1).values[:,None]



sub = (sub1 + sub2)/2
sub.iloc[:, 1:] = sub.iloc[:, 1:].values / sub.iloc[:, 1:].sum(1).values[:,None]


print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))


sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')


# =============================================================================
# submission
# =============================================================================
#if EXE_SUBMIT:
#    print('submit')
#    utils.submit(SUBMIT_FILE_PATH, COMMENT)


# =============================================================================
# CV
# =============================================================================

oof1 = pd.read_pickle(f'../data/oof_956_predict_1214-1.py.pkl')
oof1.iloc[:, :] = oof1.iloc[:, :].values / oof1.iloc[:, :].sum(1).values[:,None]

oid_gal = pd.read_pickle('../data/tr_oid_gal.pkl')['object_id'].tolist()
oid_exgal = pd.read_pickle('../data/tr_oid_exgal.pkl')['object_id'].tolist()

classes_gal = [6, 16, 53, 65, 92]
classes_exgal = [15, 42, 52, 62, 64, 67, 88, 90, 95]


sub_tr = utils.load_train(['object_id'])

sub_tr = pd.concat([sub_tr, oof1], axis=1)
sub_tr.columns = ['object_id'] +[f'class_{i}' for i in sorted(classes_gal+classes_exgal)]


sub_tr.loc[sub_tr.object_id.isin(oid_gal),  [f'class_{i}' for i in classes_exgal]] = 0
sub_tr.loc[sub_tr.object_id.isin(oid_exgal),[f'class_{i}' for i in classes_gal]] = 0

weight = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
weight = weight / sub_tr.iloc[:,1:].sum()
weight = weight.values

y_pred = sub_tr.iloc[:,1:].values.astype(float)
y_pred *= weight
sub_tr.iloc[:,1:] = y_pred

oof1 = sub_tr.iloc[:,1:].values.astype(float)



oof2 = pd.read_pickle(f'../data/oof_959_predict_1215-1.py.pkl')
oof2.iloc[:, :] = oof2.iloc[:, :].values / oof2.iloc[:, :].sum(1).values[:,None]

weight = np.array([1.1768777584208459, 0.9498970981272328, 0.6113702626667485, 0.48242068928933035, 1.2894930889416614, 1.423971561601788, 0.6535155757119984, 1.6161049089839221, 0.5743188118409728, 1.1906849086994178, 0.6527050232072442, 0.42181435682919677, 0.9394690895273552, 1.061672745432284])

classes_gal = [6, 16, 53, 65, 92]
classes_exgal = [15, 42, 52, 62, 64, 67, 88, 90, 95]


sub_tr = utils.load_train(['object_id'])

sub_tr = pd.concat([sub_tr, oof2], axis=1)
sub_tr.columns = ['object_id'] +[f'class_{i}' for i in sorted(classes_gal+classes_exgal)]


sub_tr.loc[sub_tr.object_id.isin(oid_gal),  [f'class_{i}' for i in classes_exgal]] = 0
sub_tr.loc[sub_tr.object_id.isin(oid_exgal),[f'class_{i}' for i in classes_gal]] = 0

oof2 = sub_tr.iloc[:,1:].values.astype(float) * weight


oof = (oof1 + oof2) /2

y = utils.load_target().target
print('oof1:', utils_metric.multi_weighted_logloss(y.values, oof1))
print('oof2:', utils_metric.multi_weighted_logloss(y.values, oof2))
print('ave:', utils_metric.multi_weighted_logloss(y.values, oof))















