#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:11:50 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils, utils_post
utils.start(__file__)


# =============================================================================
# weight
# =============================================================================
oof = pd.read_pickle('../FROM_MYTEAM/oof_v103_068_lgb__v103_062_nn__specz_avg.pkl')

y = utils.load_target().target
y_ohe = pd.get_dummies(y)

weight = utils_post.get_weight(y_ohe, oof, eta=0.1, nround=9999)

print(f'weight: np.array({list(weight)})')

#==============================================================================
utils.end(__file__)
