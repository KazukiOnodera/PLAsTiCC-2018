#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:39:00 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import utils

M = 14

# =============================================================================
# load
# =============================================================================
y_true = pd.read_pickle('../data/y_true.pkl').values

y_pred = pd.read_pickle('../data/oof.pkl')

weights = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])


tr = pd.read_pickle('../data/train.pkl')

idx_gal = tr[tr['hostgal_photoz'] == 0].index
idx_exgal = tr[tr['hostgal_photoz'] != 0].index


y_pred.iloc[idx_gal, [1, 3, 4, 6, 7, 9, 10, 11, 13]] = 0
y_pred.iloc[idx_exgal, [0, 2, 5, 8, 12]] = 0

y_pred = y_pred.values.astype(float)
y_pred /= y_pred.sum(1)[:,None]

for i in range(14):
    print( log_loss(y_true[:,i], y_pred[:,i]) )

tmp = y_true * y_pred

tr['loss'] = 1 - tmp.max(1)



# =============================================================================
# y_pred
# =============================================================================


#def eval_sub(y_true, y_pred, myweight=None):
#    y_pred = y_pred.copy()
#    N = y_true.shape[0]
#    if myweight is None:
#        myweight = np.ones(M)
#    for i in range(M):
#        y_pred[:,i] *= myweight[i]
#    
#    # normalize
#    y_pred /= y_pred.sum(1)[:,None]
#    
#    logloss = 0
#    for i in range(M):
#        tmp = 0
#        w = weights[i]
#        for j in range(N):
#            tmp += (y_true[j,i] * np.log( y_pred[j,i] ))
#        logloss += w * tmp / sum(y_true[:,i])
#    logloss /= -sum(weights)
#    return logloss
#
#
#eval_sub(y_true.values, y_pred.values.astype(float))
#
#eval_sub(y_true, y_pred, weights)

# =============================================================================
# 
# =============================================================================

def multi_weighted_logloss(y_true, y_pred, myweight=None):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    
    if myweight is None:
        myweight = np.ones(M)
    y_p = y_pred * myweight
    
    # normalize
    y_p /= y_p.sum(1)[:,None]
    
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_true * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = np.nansum(y_true, axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.nansum(y_w) / np.sum(class_arr)
    return loss


def calc_gradient(f, X):
    """
    calc_gradient
    偏微分を行う関数
    関数fを変数xの各要素で偏微分した結果をベクトルにした勾配を返す
    
    @params
    f: 対象となる関数
    X: 関数fの引数のベクトル(numpy.array)
    
    @return
    gradient: 勾配(numpy.array)
    """
    
    h = 1e-4
    gradient = np.zeros_like(X)
    
    # 各変数についての偏微分を計算する
    for i in range(X.size):
        store_X = X[:]
        
        # f(x+h)
        X[i] += h
        f_x_plus_h = f(X)

        X = store_X[:]
        
        # f(x-h)
        X[i] -= h
        f_x_minus_h = f(X)
        
        # 偏微分
        gradient[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)
        
    return gradient

def gradient_descent(f, X, learning_rate, max_iter):
    """
    gradient_descent
    最急降下法を行う関数
    
    @params
    f: 対象となる関数
    X: 関数fの引数のベクトル(numpy.array)
    learning_rate: 学習率
    max_iter: 繰り返し回数
    
    @return
    X: 関数の出力を最小にする(であろう)引数(numpy.array)
    """
    
    for i in range(max_iter):
        X -= (learning_rate * calc_gradient(f, X))
        if i%50==0:
            print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, f(X)))
        
    return X

def validation(X_train, X_test, y_train, y_test, learning_rate=0.1, max_iter=9999):
    weight = np.ones(X_train.shape[1])
    f1 = lambda X: multi_weighted_logloss(X_train, y_train, weight)
    f2 = lambda X: multi_weighted_logloss(X_test, y_test, weight)
    
    for i in range(max_iter):
        weight -= (learning_rate * calc_gradient(f1, weight))
        if i%50==0:
            print(f'[{i:3d}] train: {f1(weight):.7f}    test: {f2(weight):.7f}')
    
    return

# =============================================================================
# 
# =============================================================================

multi_weighted_logloss(y_true, y_pred, myweight=None)
multi_weighted_logloss(y_true[:888], y_pred[:888], myweight=None)

t_train, t_test, p_train, p_test = train_test_split(y_true, y_pred,
                                                    test_size=0.33, 
                                                    random_state=42)


multi_weighted_logloss(t_train, p_train, myweight=None)
multi_weighted_logloss(t_test, p_test, myweight=None)

validation(t_train, t_test, p_train, p_test)





f = lambda X: multi_weighted_logloss(y_true, y_pred, X)

X = np.ones(M)
X = weights.copy()

gradient_descent(f, X, learning_rate=0.5, max_iter=1000)


y_pred * X


