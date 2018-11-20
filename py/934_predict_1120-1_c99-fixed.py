#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 00:26:32 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils

utils.start(__file__)



SUBMIT_FILE_PATH = '../output/1120-1_c99-fix.csv.gz'

COMMENT = '1120-1 + fixed class99'


sub = pd.read_pickle('../data/y_pred_raw_934_predict_1120-1.py.pkl')

def class99_stat_process(softmax_array, stat99prob=1/9):
    return np.hstack((softmax_array*(1-stat99prob), np.ones((softmax_array.shape[0],1))*stat99prob))


sub_ = class99_stat_process(sub.iloc[:,1:].values)

sub_ = pd.DataFrame(sub_, columns=sub.iloc[:,1:].columns.tolist()+['class_99'])
sub_['object_id'] = sub['object_id']

sub_.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
print('submit')
utils.submit(SUBMIT_FILE_PATH, COMMENT)


utils.end(__file__)

