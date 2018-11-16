#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:22:37 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm

import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count

import torch
#import torch.nn.functional as F
from torch.autograd import grad

import utils
#utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)

DROP = ['f001_hostgal_specz', 'f001_distmod', 'f701_hostgal_specz'] # 

#DROP = []

NFOLD = 5

LOOP = 1

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
#         'learning_rate': 0.01,
         'max_depth': 3,
#         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
#         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }

# =============================================================================
# def
# =============================================================================
classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1,
                64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
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


# =============================================================================
# load
# =============================================================================

files_tr = sorted(glob('../data/train_f*.pkl'))
[print(f) for f in files_tr]

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
y = utils.load_target().target

X.drop(DROP, axis=1, inplace=True)

target_dict = {}
target_dict_r = {}
for i,e in enumerate(y.sort_values().unique()):
    target_dict[e] = i
    target_dict_r[i] = e

y = y.replace(target_dict)

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

#CAT = list( set(X.columns)&set(utils_cat.ALL))
#print(f'CAT: {CAT}')

# =============================================================================
# cv1
# =============================================================================
param['learning_rate'] = 0.1
dtrain = lgb.Dataset(X, y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
for i in range(LOOP):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         feval=utils.lgb_multi_weighted_logloss,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/LOOP) * 1.3)

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)

utils.send_line(result)
imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

print(imp.head(200).feature.map(lambda x: x.split('_')[0]).value_counts())

imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

"""

__file__ = '801_cv.py'
imp = pd.read_csv(f'LOG/imp_{__file__}-1.csv')

"""



# =============================================================================
# cv2
# =============================================================================
COL = imp.feature.tolist()[:3000]

param['learning_rate'] = 0.1
dtrain = lgb.Dataset(X[COL], y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
for i in range(LOOP):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                            fobj=wloss_objective, 
                            feval=wloss_metric,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/LOOP) * 1.3)

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)

utils.send_line(result)
imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

print(imp.head(200).feature.map(lambda x: x.split('_')[0]).value_counts())

imp.to_csv(f'LOG/imp_{__file__}-2.csv', index=False)



# =============================================================================
# estimate feature size
# =============================================================================
print('estimate feature size')
param['learning_rate'] = 0.1

COL = imp.feature.tolist()
best_score = 9999
best_N = 0

for i in np.arange(100, 500, 50):
    print(f'\n==== feature size: {i} ====')
    
    dtrain = lgb.Dataset(X[COL[:i]], y, #categorical_feature=CAT, 
                         free_raw_data=False)
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         fobj=wloss_objective, 
                         feval=wloss_metric,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    
    score = ret['wloss-mean'][-1]
    utils.send_line(f"feature size: {i}    wloss-mean: {score}")
    
    if score < best_score:
        best_score = score
        best_N = i



# =============================================================================
# best
# =============================================================================
N = best_N
dtrain = lgb.Dataset(X[COL[:N]], y, #categorical_feature=CAT, 
                     free_raw_data=False)
ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                     feval=utils.lgb_multi_weighted_logloss,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=SEED)

score = ret['wloss-mean'][-1]

y_pred = ex.eval_oob(X[COL[:N]], y, models, SEED, stratified=True, shuffle=True, 
                     n_class=y.unique().shape[0])

tmp = y_pred.copy()

y_pred = pd.DataFrame(softmax(y_pred.astype(float).values))


y_true = pd.get_dummies(y)


y_true.to_pickle('../data/y_true.pkl')
y_pred.to_pickle('../data/y_pred.pkl')


# =============================================================================
# confusion matrix
# =============================================================================
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import itertools

unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i
        
y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'LOG/CM_{__file__}.png')



cnf_matrix = confusion_matrix(y_map, np.argmax(y_pred.values, axis=-1))
np.set_printoptions(precision=2)

class_names = ['class_6',
             'class_15',
             'class_16',
             'class_42',
             'class_52',
             'class_53',
             'class_62',
             'class_64',
             'class_65',
             'class_67',
             'class_88',
             'class_90',
             'class_92',
             'class_95']

foo = plot_confusion_matrix(cnf_matrix, classes=class_names,
                            normalize=True,
                            title='Confusion Matrix')

utils.send_line(f'Confusion Matrix wmlogloss: {score}', png=f'LOG/CM_{__file__}.png')



#==============================================================================
#utils.end(__file__)
#utils.stop_instance()


