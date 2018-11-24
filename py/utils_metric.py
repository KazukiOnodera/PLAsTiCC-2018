#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 23:39:22 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os

import torch
import torch.nn.functional as F
from torch.autograd import grad

import utils

classes_gal = [6, 16, 53, 65, 92]

class_weight_gal = {6: 1, 
                    16: 1, 
                    53: 1, 
                    65: 1, 
                    92: 1}

classes_exgal = [15, 42, 52, 62, 64, 67, 88, 90, 95]

class_weight_exgal = {15: 2, 
                        42: 1, 
                        52: 1, 
                        62: 1, 
                        64: 2, 
                        67: 1, 
                        88: 1, 
                        90: 1, 
                        95: 1}

classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]

class_weight = {6: 1, 
                15: 2, 
                16: 1, 
                42: 1, 
                52: 1, 
                53: 1, 
                62: 1, 
                64: 2, 
                65: 1, 
                67: 1, 
                88: 1, 
                90: 1, 
                92: 1, 
                95: 1}

def lgb_multi_weighted_logloss_gal(y_preds, train_data):
    """
    @author olivier https://www.kaggle.com/ogrellier
    https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data/code
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    y_true = train_data.get_label()
    if len(np.unique(y_true)) > 14:
        classes_gal.append(99)
        class_weight_gal[99] = 2
    
    y_p = y_preds.reshape(y_true.shape[0], len(classes_gal), order='F')
    # normalize
    y_p /= y_p.sum(1)[:,None]

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight_gal[k] for k in sorted(class_weight_gal.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False

def lgb_multi_weighted_logloss_exgal(y_preds, train_data):
    """
    @author olivier https://www.kaggle.com/ogrellier
    https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data/code
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    y_true = train_data.get_label()
    if len(np.unique(y_true)) > 14:
        classes_exgal.append(99)
        class_weight_exgal[99] = 2
    
    y_p = y_preds.reshape(y_true.shape[0], len(classes_exgal), order='F')
    # normalize
    y_p /= y_p.sum(1)[:,None]

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight_exgal[k] for k in sorted(class_weight_exgal.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False

def lgb_multi_weighted_logloss(y_preds, train_data):
    """
    @author olivier https://www.kaggle.com/ogrellier
    https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data/code
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    y_true = train_data.get_label()
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False

def multi_weighted_logloss(y_true:np.array, y_preds:np.array):
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
        
    y_p = y_preds/y_preds.sum(1)[:,None]
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

# =============================================================================
# 
# =============================================================================

weight_tensor = torch.tensor(list(class_weight.values()),
                             requires_grad=False).type(torch.FloatTensor)
class_dict = {c: i for i, c in enumerate(classes)}

# this is a reimplementation of the above loss function using pytorch expressions.
# Alternatively this can be done in pure numpy (not important here)
# note that this function takes raw output instead of probabilities from the booster
# Also be aware of the index order in LightDBM when reshaping (see LightGBM docs 'fobj')
def wloss_metric(preds, train_data):
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    y_h /= y_h.sum(dim=0, keepdim=True)
    y_p = torch.tensor(preds, requires_grad=False).type(torch.FloatTensor)
    if len(y_p.shape) == 1:
        y_p = y_p.reshape(len(classes), -1).transpose(0, 1)
    ln_p = torch.log_softmax(y_p, dim=1)
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll) / torch.sum(weight_tensor)
    return 'wloss', loss.numpy() * 1., False

def wloss_objective(preds, train_data):
    y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
    y_h = torch.zeros(
        y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
    ys = y_h.sum(dim=0, keepdim=True)
    y_h /= ys
    y_p = torch.tensor(preds, requires_grad=True).type(torch.FloatTensor)
    y_r = y_p.reshape(len(classes), -1).transpose(0, 1)
    ln_p = torch.log_softmax(y_r, dim=1)
    wll = torch.sum(y_h * ln_p, dim=0)
    loss = -torch.dot(weight_tensor, wll)
    grads = grad(loss, y_p, create_graph=True)[0]
    grads *= float(len(classes)) / torch.sum(1 / ys)  # scale up grads
    hess = torch.ones(y_p.shape)  # haven't bothered with properly doing hessian yet
    return grads.detach().numpy(), \
        hess.detach().numpy()

#import tensorflow as tf
#tf.enable_eager_execution()
#tfe = tf.contrib.eager
#
#weight_tensor = tf.convert_to_tensor(list(class_weight.values()), dtype=tf.float32)
#
#def wloss_metric(preds, train_data):
#    y_t = tf.convert_to_tensor(train_data.get_label())
#    y_h = tf.one_hot(y_t, depth=14, dtype=tf.float32)
#    y_h /= tf.reduce_sum(y_h, axis=0, keepdims=True)
#    y_p = tf.convert_to_tensor(preds, dtype=tf.float32)
#    if len(y_p.shape) == 1:
#        y_p = tf.transpose(tf.reshape(y_p, (len(classes), -1)), perm=(1, 0))
##     ln_p = tf.nn.log_softmax(y_p, axis=1)
#    ln_p = tf.log(tf.clip_by_value(tf.nn.softmax(y_p, axis=1), 1e-15, 1-1e-15))
#    wll = tf.reduce_sum(y_h * ln_p, axis=0)
#    loss = -tf.reduce_sum(weight_tensor * wll) / tf.reduce_sum(weight_tensor)
#    return 'wloss', loss.numpy(), False
#
#def grad(f):
#    return lambda x: tfe.gradients_function(f)(x)[0]
#
#def wloss_objective(preds, train_data):
#    y_t = tf.convert_to_tensor(train_data.get_label())
#    y_h = tf.one_hot(y_t, depth=14, dtype=tf.float32)
#    ys = tf.reduce_sum(y_h, axis=0, keepdims=True)
#    y_h /= ys
#    y_p = tf.convert_to_tensor(preds, dtype=tf.float32)
#    def loss(y_p):
#        if len(y_p.shape) == 1:
#            y_p = tf.transpose(tf.reshape(y_p, (len(classes), -1)), perm=(1, 0))
#        ln_p = tf.nn.log_softmax(y_p, axis=1)
##         ln_p = tf.log(tf.clip_by_value(tf.nn.softmax(y_p, axis=1), 1e-15, 1-1e-15))
#        wll = tf.reduce_sum(y_h * ln_p, axis=0)
#        return -tf.reduce_sum(weight_tensor * wll) * len(train_data.get_label())
#    grads = grad(loss)(y_p)
##     hess = grad(grad(loss))(y_p)
##     hess /= tf.reduce_mean(hess)
#    hess = tf.ones(y_p.shape)
#    return grads.numpy(), hess.numpy()


def akiyama_metric(y_true, y_preds):
    '''
    y_true:１次元のnp.array
    y_pred:softmax後の１4次元のnp.array
    '''
    class99_prob = 1/9
    class99_weight = 2
            
    y_p = y_preds * (1-class99_prob)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    
    y_true_ohe = pd.get_dummies(y_true).values
    nb_pos = y_true_ohe.sum(axis=0).astype(float)
    
#    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 
                    67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    
    y_log_ones = np.sum(y_true_ohe * y_p_log, axis=0)
    y_w = y_log_ones * class_arr / nb_pos
    score = - np.sum(y_w) / (np.sum(class_arr)+class99_weight)\
        + (class99_weight/(np.sum(class_arr)+class99_weight))*(-np.log(class99_prob))

    return score

def softmax(x, axis=1):
    z = np.exp(x)
    return z / np.sum(z, axis=axis, keepdims=True)



