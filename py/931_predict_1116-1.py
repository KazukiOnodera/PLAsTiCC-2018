#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:22:51 2018

@author: kazuki.onodera
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
utils.start(__file__)
#==============================================================================

SUBMIT_FILE_PATH = '../output/1116-1.csv.gz'

COMMENT = 'top300 features'

EXE_SUBMIT = True

#DROP = ['f001_hostgal_specz']

SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5

LOOP = 5

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.1,
         'max_depth': 3,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }

N_FEATURES = 400

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
COL = pd.read_csv('LOG/imp_801_cv.py-2.csv').head(N_FEATURES).feature.tolist()

PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)[COL]
y = utils.load_target().target

#X.drop(DROP, axis=1, inplace=True)

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
# cv
# =============================================================================

dtrain = lgb.Dataset(X, y, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
y_preds = []
for i in range(2):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         fobj=wloss_objective, feval=wloss_metric,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    y_pred = ex.eval_oob(X, y, models, SEED, stratified=True, shuffle=True, 
                         n_class=y.unique().shape[0])
    y_preds.append(y_pred)
    model_all += models
    nround_mean += len(ret['wloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/2) * 1.3)

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)

imp = ex.getImp(model_all)
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

png = f'LOG/imp_{__file__}.png'
utils.savefig_imp(imp, png, x='total', title=f'{__file__}')
utils.send_line(result, png)

# =============================================================================
# weight
# =============================================================================
import utils_post

for i,y_pred in enumerate(y_preds):
    if i==0:
        tmp = y_pred
    else:
        tmp += y_pred
tmp /= len(y_preds)
tmp.to_pickle('../data/oof.pkl')

tmp = tmp.values.astype(float)

y_true = pd.get_dummies(y)

weight = utils_post.get_weight(y_true, tmp, eta=0.1, nround=9999)
weight = np.append(weight, 1)
print(list(weight))

# =============================================================================
# model
# =============================================================================

#gc.collect()
#
#
#np.random.seed(SEED)
#
#model_all = []
#for i in range(LOOP):
#    print('building', i)
#    gc.collect()
#    param['seed'] = np.random.randint(9999)
#    model = lgb.train(param, dtrain, num_boost_round=nround_mean, valid_sets=None, 
#                      valid_names=None, fobj=None, feval=None, init_model=None, 
#                      feature_name='auto', categorical_feature='auto', 
#                      early_stopping_rounds=None, evals_result=None, 
#                      verbose_eval=True, learning_rates=None, 
#                      keep_training_booster=False, callbacks=None)
#    
#    model_all.append(model)
#
#
#del dtrain, X; gc.collect()

# =============================================================================
# test
# =============================================================================


files_te = []
for pref in PREFS:
    files_te += glob(f'../data/test_{pref}*.pkl')


def read(f):
    df = pd.read_pickle(f)
    col = list( set(df.columns) & set(COL) )
    return df[col]

X_test = pd.concat([
                read(f) for f in tqdm(files_te, mininterval=60)
               ], axis=1)[COL]

gc.collect()
    
for i,model in enumerate(tqdm(model_all)):
    gc.collect()
    y_pred = model.predict(X_test)
    if i==0:
        y_pred_all = y_pred
    else:
        y_pred_all += y_pred

y_pred_all /= len(model_all)

sub = pd.read_csv('../input/sample_submission.csv.zip')
df = pd.DataFrame(y_pred_all, columns=sub.columns[1:-1])

sub = pd.concat([sub[['object_id']], df], axis=1)

# class_99
sub.to_pickle('../data/y_pred_raw.pkl')
utils.postprocess(sub, method='oli')
#utils.postprocess(sub, weight=weight, method='giba')

print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))


sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')


png = f'LOG/sub_{__file__}.png'
utils.savefig_sub(sub, png)
utils.send_line('DONE!', png)

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)

#==============================================================================
utils.end(__file__)
utils.stop_instance()


