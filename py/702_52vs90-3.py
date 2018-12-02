#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:50:44 2018

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

import utils
utils.start(__file__)
#==============================================================================

PREF = 'f702'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')



SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5

LOOP = 5

param = {
         'objective': 'binary',
         'metric': 'auc',
#         'metric': 'binary_logloss',
         
         'learning_rate': 0.01,
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


BASE_FEATURES = 150


# =============================================================================
# load
# =============================================================================
COL = pd.read_csv('LOG/imp_702_52vs90.py-2.csv').head(BASE_FEATURES).feature.tolist()

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

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

# =============================================================================
# subset
# =============================================================================
print('subset')

X_52_90 = X[ y.isin([target_dict[52], target_dict[90],])]
X_oth   = X[~y.isin([target_dict[52], target_dict[90],])]
y_52_90 = y[ y.isin([target_dict[52], target_dict[90],])]

y_52_90 = (y_52_90==target_dict[52])*1

print(f'X_52_90.shape {X_52_90.shape}')

X_oid       = utils.load_train(['object_id', 'target'])
X_oid_52_90 = X_oid[ y.isin([target_dict[52], target_dict[90],])]
X_oid_oth   = X_oid[~y.isin([target_dict[52], target_dict[90],])]

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X_52_90, y_52_90.values,
                     free_raw_data=False)

gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
oofs = []
for i in range(LOOP):
    
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    y_pred = ex.eval_oob(X_52_90, y_52_90.values, models, SEED, 
                         stratified=True, shuffle=True, 
                         )
    oofs.append(y_pred)
    model_all += models
    nround_mean += len(ret['auc-mean'])
    wloss_list.append( ret['auc-mean'][-1] )

nround_mean = int((nround_mean/2) * 1.3)
utils.send_line(f'nround_mean: {nround_mean}')

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
utils.send_line(result)


for i,y_pred in enumerate(oofs):
    if i==0:
        tmp = y_pred
    else:
        tmp += y_pred
tmp /= len(oofs)
oof = tmp.copy().values.astype(float)

X_oid_52_90['52vs90'] = oof

# =============================================================================
# predict other train
# =============================================================================


for i,model in enumerate(tqdm(model_all)):
    gc.collect()
    y_pred = model.predict(X_oth)
    if i==0:
        y_pred_all = y_pred
    else:
        y_pred_all += y_pred

y_pred_all /= len(model_all)

X_oid_oth['52vs90'] = y_pred_all

del dtrain, X; gc.collect()

X_train = pd.concat([X_oid_52_90, X_oid_oth], 
                    ignore_index=True).sort_values('object_id').reset_index(drop=True)[['52vs90']]

X_train.add_prefix(PREF+'_').to_pickle(f'../data/train_{PREF}.pkl')



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
    if i==0:
        y_pred_all = y_pred
    else:
        y_pred_all += y_pred

y_pred_all /= len(model_all)



X_test = pd.DataFrame(y_pred_all, columns=['52vs90']).add_prefix(PREF+'_')
utils.to_pkl_gzip(X_test, f'../data/test_{PREF}.pkl')


#==============================================================================
utils.end(__file__)
#utils.stop_instance()


