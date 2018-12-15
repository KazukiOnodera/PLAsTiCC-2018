#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:25:09 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
#import os
#from multiprocessing import cpu_count, Pool
#import utils


def multi_weighted_logloss(y_true, y_pred, myweight=None, based_true=True):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if y_true.shape[1] > 14:
        classes.append(99)
        class_weight[99] = 2
    
    if myweight is None:
        myweight = np.ones(y_true.shape[1])
    y_p = y_pred * myweight
    
    # normalize
    y_p /= y_p.sum(1)[:,None]
    
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=0, a_max=1)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_true * y_p_log, axis=0)
    
    # Get the number of positives for each class
    if based_true == True:
        nb_pos = y_true.sum(axis=0).astype(float)
    else:
        nb_pos = pd.DataFrame(y_pred).sum(axis=0).astype(float)
        
    
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

def gradient_descent(f, X, learning_rate, max_iter, is_print=True, verbose_eval=100):
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
    sw_break = False
    score_bk = 9999
    for i in range(max_iter):
        X -= (learning_rate * calc_gradient(f, X))
        score = f(X)
        
        if score_bk <= score:
            sw_break = True
            break
        score_bk = score
        
        if is_print and i%verbose_eval==0:
            print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, score))
    
    if is_print and sw_break:
        print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, score))
    
    return X

def get_weight(y_true, y_pred, weight=None, eta=1, nround=100, is_print=True, verbose_eval=50):
    M = y_true.shape[1]
    if weight is None:
        weight = np.ones(M)
    f = lambda X: multi_weighted_logloss(y_true, y_pred, weight)
    gradient_descent(f, weight, learning_rate=eta, max_iter=nround,
                     is_print=is_print, verbose_eval=verbose_eval)
    return weight




