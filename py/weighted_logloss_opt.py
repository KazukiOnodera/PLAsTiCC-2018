#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:39:00 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd

M = 5
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



def eval_sub(y_true, y_pred, myweight=None):
    y_pred = y_pred.copy()
    
    if myweight is None:
        myweight = np.ones(M)
    for i in range(M):
        y_pred[:,i] *= myweight[i]
    
    # normalize
    sum1 = y_pred.sum(1)
    for i in range(M):
        y_pred[:,i] /= sum1
    
    logloss = 0
    for i in range(M):
        tmp = 0
        w = weights[i]
        for j in range(N):
            tmp += (y_true[j,i] * np.log( y_pred[j,i] ))
        logloss += w * tmp / sum(y_true[:,i])
    logloss /= -sum(weights)
    return logloss

y_true.sum(0)
#y_pred[:,1] *= 2
eval_sub(y_true, y_pred)

eval_sub(y_true, y_pred, weights)

# =============================================================================
# 
# =============================================================================
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
        print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, f(X)))
        
    return X

f = lambda X: eval_sub(y_true, y_pred, X)

X = np.ones(M)
X = weights.copy()

gradient_descent(f, X, learning_rate=0.51, max_iter=100)

