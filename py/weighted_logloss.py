#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:18:46 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd


y_true = np.random.rand(100, 10)

argmax = y_true.argmax(1)
for i,e in enumerate(argmax):
    y_true[i] = 0
    y_true[i, e] = 1


weights = np.random.uniform(size=10)


y_pred = np.random.rand(100, 10)
sum1 = y_pred.sum(1)
for i in range(10):
    y_pred[:,i] /= sum1



logloss = 0
M = len(weights)
N = y_true.shape[0]
for i in range(M):
    tmp = 0
    w = weights[i]
    for j in range(N):
        tmp += (y_true[j,i] * np.log(y_pred[j,i]))
    logloss += w * tmp / sum(y_true[:,i])
logloss /= -sum(weights)


