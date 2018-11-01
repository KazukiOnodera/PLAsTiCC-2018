#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:39:00 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd

M = 10
N = 9999

# =============================================================================
# y_true1
# =============================================================================
y_true = np.random.randn(N, M)

argmax = y_true.argmax(1)
for i,e in enumerate(argmax):
    y_true[i] = 0
    y_true[i, e] = 1

weights = np.random.uniform(size=M)


# =============================================================================
# y_true2
# =============================================================================
y_true = np.random.randn(N, M)

for i in range(M):
    if np.random.uniform()>0.5:
        y_true[:, i] *= 3

argmax = y_true.argmax(1)
for i,e in enumerate(argmax):
    y_true[i] = 0
    y_true[i, e] = 1

weights = np.random.uniform(size=M)

# =============================================================================
# y_pred
# =============================================================================
tmp = np.random.uniform(size=M)
y_pred = np.array([tmp for i in range(N)])



def eval_sub(y_true, y_pred):
    y_pred = y_pred.copy()
    sum1 = y_pred.sum(1)
    for i in range(M):
        y_pred[:,i] /= sum1
    
    logloss = 0
    for i in range(M):
        tmp = 0
        w = weights[i]
        for j in range(N):
            tmp += (y_true[j,i] * np.log(y_pred[j,i]))
        logloss += w * tmp / sum(y_true[:,i])
    logloss /= -sum(weights)
    return logloss

y_true.sum(0)
#y_pred[:,1] *= 2
eval_sub(y_true, y_pred)


