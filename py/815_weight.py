#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:11:50 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils, utils_post, utils_metric
utils.start(__file__)


# =============================================================================
# weight
# =============================================================================
oof = pd.read_pickle('../FROM_MYTEAM/oof_v103_068_lgb__v103_062_nn__specz_avg.pkl')
oof = oof.copy().values.astype(float)

y = utils.load_target().target
y_ohe = pd.get_dummies(y)

weight = utils_post.get_weight(y_ohe, oof, eta=0.1, nround=9999)

print(f'weight: np.array({list(weight)})')



# =============================================================================
# one by one
# =============================================================================


oof = pd.read_pickle('../FROM_MYTEAM/oof_v103_068_lgb__v103_062_nn__specz_avg.pkl')

oid_gal = pd.read_pickle('../data/tr_oid_gal.pkl')['object_id'].tolist()
oid_exgal = pd.read_pickle('../data/tr_oid_exgal.pkl')['object_id'].tolist()

classes_gal = [6, 16, 53, 65, 92]
classes_exgal = [15, 42, 52, 62, 64, 67, 88, 90, 95]


sub_tr = utils.load_train(['object_id'])

sub_tr = pd.concat([sub_tr, oof], axis=1)
sub_tr.columns = ['object_id'] +[f'class_{i}' for i in sorted(classes_gal+classes_exgal)]


sub_tr.loc[sub_tr.object_id.isin(oid_gal),  [f'class_{i}' for i in classes_exgal]] = 0
sub_tr.loc[sub_tr.object_id.isin(oid_exgal),[f'class_{i}' for i in classes_gal]] = 0

#weight = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
#weight = weight / sub_tr.iloc[:,1:].sum()
#weight = weight.values
#
oof = sub_tr.iloc[:,1:].values.astype(float)

y = utils.load_target().target
y_ohe = pd.get_dummies(y)

oof_aug = np.array([oof[0] for i in range(9999)])



#==============================================================================
utils.end(__file__)
