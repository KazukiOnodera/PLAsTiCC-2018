#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:39:29 2018

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

import utils, utils_metric
utils.start(__file__)
#==============================================================================

SUBMIT_FILE_PATH = '../output/1129-1.csv.gz'

COMMENT = 'top100 features * 10'

EXE_SUBMIT = True


SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5

LOOP = 10

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.5,
         'max_depth': 3,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.7,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }


BASE_FEATURES = 100


# =============================================================================
# load
# =============================================================================
COL = pd.read_csv('LOG/imp_801_cv.py-2.csv').head(BASE_FEATURES).feature.tolist()

PREFS = sorted(set([c.split('_')[0] for c in COL]))

files_tr = []
for pref in PREFS:
    files_tr += glob(f'../data/train_{pref}*.pkl')

files_te = [f'../feature/test_{c}.pkl' for c in COL]
sw = False
for i in files_te:
    if os.path.exists(i)==False:
        print(i)
        sw = True
if sw:
    raise Exception()

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


# drop 100 outliers
drop_index = pd.Index([3446, 1601, 6684, 6969, 2859, 6615, 7312, 1707, 6337, 6084, 2702,
             3690,  774, 7763, 7694, 5530,  376, 4465, 4114, 3795, 7128, 6557,
             3611, 2983, 3369, 3037, 2881, 2132, 2575, 2611, 7654, 3679, 5678,
             4590, 5981, 3426, 4151, 2393, 6530, 2449, 6665, 2784, 2832, 3998,
             5119, 7384, 5876, 5566, 4142, 4695, 6360, 3314, 4218, 7111, 2834,
             3287, 3906, 3282, 3267, 5748, 6481, 4423, 3549, 2250, 4333, 6370,
             4708, 5170, 7283, 3645, 7559, 3825, 4140, 3194, 5369, 2656, 6975,
             4740, 4387, 6374, 3118, 2638, 6249, 2452, 6747, 2224, 3917, 5944,
             4070, 5098, 3271, 4645, 3493, 5696,  333, 6723, 6749, 1064, 2969,
             7357])

X = X.drop(drop_index).reset_index()
y = y.drop(drop_index).reset_index()



if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X[COL], y.values, #categorical_feature=CAT, 
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
                         fobj=utils_metric.wloss_objective, 
                         feval=utils_metric.wloss_metric,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    y_pred = ex.eval_oob(X[COL], y.values, models, SEED, stratified=True, shuffle=True, 
                         n_class=True)
    y_preds.append(y_pred)
    model_all += models
    nround_mean += len(ret['wloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/2) * 1.3)
utils.send_line(f'nround_mean: {nround_mean}')

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)
utils.send_line(result)


for i,y_pred in enumerate(y_preds):
    y_pred = pd.DataFrame(utils_metric.softmax(y_pred.astype(float).values))
    if i==0:
        tmp = y_pred
    else:
        tmp += y_pred
tmp /= len(y_preds)
y_preds = tmp.copy().values.astype(float)

a_score = utils_metric.akiyama_metric(y.values, y_preds)
print(f'akiyama_metric: {a_score}')

utils.send_line(f'akiyama_metric: {a_score}')


# =============================================================================
# model
# =============================================================================

gc.collect()


np.random.seed(SEED)

model_all = []
for i in range(LOOP):
    
    print(f'LOOP:{i}')
    gc.collect()
    param['seed'] = np.random.randint(9999)
    model = lgb.train(param, dtrain, num_boost_round=nround_mean, 
                      fobj=utils_metric.wloss_objective, 
                      feval=utils_metric.wloss_metric,
                      valid_names=None, init_model=None, 
                      feature_name='auto', categorical_feature='auto', 
                      early_stopping_rounds=None, evals_result=None, 
                      verbose_eval=True, learning_rates=None, 
                      keep_training_booster=False, callbacks=None)

    model_all.append(model)


del dtrain, X; gc.collect()

# =============================================================================
# test
# =============================================================================

X_test = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_te, mininterval=10)
               ], axis=1)[COL]

gc.collect()

for i,model in enumerate(tqdm(model_all)):
    gc.collect()
    y_pred = model.predict(X_test)
    y_pred = utils_metric.softmax(y_pred)
    if i==0:
        y_pred_all = y_pred
    else:
        y_pred_all += y_pred

y_pred_all /= len(model_all)


sub = pd.read_csv('../input/sample_submission.csv.zip')
df = pd.DataFrame(y_pred_all, columns=sub.columns[1:-1])

sub = pd.concat([sub[['object_id']], df], axis=1)

# class_99
sub.to_pickle(f'../data/y_pred_raw_{__file__}.pkl')
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
#utils.stop_instance()





