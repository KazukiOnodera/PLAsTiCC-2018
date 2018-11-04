#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:46:23 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count, Pool
import utils

FILE_in = '../output/1103-2.csv.gz'
FILE_out = '../output/1103-2_post.csv.gz'

M = 15

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
    if True:
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
    nb_pos = y_true.sum(axis=0).astype(float)
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

def gradient_descent(f, X, learning_rate, max_iter, is_print=False):
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
    score_bk = 9999
    for i in range(max_iter):
        X -= (learning_rate * calc_gradient(f, X))
        score = f(X)
        if score_bk < score:
            break
        score_bk = score
        if is_print and i%50==0:
            print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, score))
        
    return X

def multi(idx):
    w = np.ones(M)
    y_true = np.array( [di[y_preds[idx][i].round(2)] for i in range(M)]).T
    y_pred = np.array([y_preds[idx] for i in range(1000)])
    f = lambda X: multi_weighted_logloss(y_true, y_pred, w)
    gradient_descent(f, w, learning_rate=2, max_iter=1000)
    return y_preds[idx] * w

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    sub = pd.read_csv(FILE_in)
    y_preds = sub.iloc[:, 1:].values
    
    di = {}
    for i in np.arange(0.00, 1.01, 0.01):
        di[round(i, 2)] = np.append(np.ones(int(i*1000)), np.zeros( 1000-int(i*1000) ))
    
    pool = Pool(cpu_count())
    callback = pool.map(multi, range(y_preds.shape[0]))
    pool.close()
    
    arr = np.array(callback)
    arr /= arr.sum(1)[:,None]
    
    pd.DataFrame(arr).to_pickle('../output/tmp.pkl')
    sub.iloc[:, 1:] = arr
    sub.to_csv(FILE_out, index=False, compression='gzip')
    
    utils.end(__file__)

