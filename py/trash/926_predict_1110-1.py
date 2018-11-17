#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 23:55:02 2018

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

SUBMIT_FILE_PATH = '../output/1110-1.csv.gz'

COMMENT = 'galactic cut'

EXE_SUBMIT = True

SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)


NFOLD = 5

LOOP = 5

param = {
         'objective': 'multiclass',
#         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.01,
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.5,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }

N_FEATURES_gal   = 150
N_FEATURES_exgal = 250


# =============================================================================
# load
# =============================================================================
COL_gal   = pd.read_csv('LOG/imp_802_cv_separate.py_gal.csv').head(N_FEATURES_gal).feature.tolist()
COL_exgal = pd.read_csv('LOG/imp_802_cv_separate.py_exgal.csv').head(N_FEATURES_exgal).feature.tolist()

files_tr = sorted(glob('../data/train_f*.pkl'))
[print(f) for f in files_tr]

X = pd.concat([
                pd.read_pickle(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
y = utils.load_target().target


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

#CAT = list( set(X.columns)&set(utils_cat.ALL))
#print(f'CAT: {CAT}')

# =============================================================================
# separate
# =============================================================================

is_gal = pd.read_pickle('../data/tr_is_gal.pkl')

X_gal = X[is_gal][COL_gal]
y_gal = y[is_gal]

X_exgal = X[~is_gal][COL_exgal]
y_exgal = y[~is_gal]


def target_replace(y):
    target_dict = {}
    target_dict_r = {}
    for i,e in enumerate(y.sort_values().unique()):
        target_dict[e] = i
        target_dict_r[i] = e
    
    return y.replace(target_dict), target_dict_r


y_gal, di_gal = target_replace(y_gal)
y_exgal, di_exgal = target_replace(y_exgal)


del X, y
gc.collect()

print(f'X_gal.shape: {X_gal.shape}')
print(f'X_exgal.shape: {X_exgal.shape}')



# =============================================================================
# cv(gal)
# =============================================================================
print('==== GAL ====')
param['num_class'] = 5

dtrain = lgb.Dataset(X_gal, y_gal, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
y_preds_gal = []
for i in range(2):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         feval=utils.lgb_multi_weighted_logloss_gal,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    y_pred = ex.eval_oob(X_gal, y_gal, models, SEED, stratified=True, shuffle=True, 
                         n_class=y_gal.unique().shape[0])
    y_preds_gal.append(y_pred)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/2) * 1.3)

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)

# =============================================================================
# model(gal)
# =============================================================================

gc.collect()


np.random.seed(SEED)

models_gal = []
for i in range(LOOP):
    print('building', i)
    gc.collect()
    param['seed'] = np.random.randint(9999)
    model = lgb.train(param, dtrain, num_boost_round=nround_mean, valid_sets=None, 
                      valid_names=None, fobj=None, feval=None, init_model=None, 
                      feature_name='auto', categorical_feature='auto', 
                      early_stopping_rounds=None, evals_result=None, 
                      verbose_eval=True, learning_rates=None, 
                      keep_training_booster=False, callbacks=None)
    
    models_gal.append(model)


del dtrain; gc.collect()




# =============================================================================
# cv(exgal)
# =============================================================================
print('==== EXGAL ====')
param['num_class'] = 9

dtrain = lgb.Dataset(X_exgal, y_exgal, #categorical_feature=CAT, 
                     free_raw_data=False)
gc.collect()

model_all = []
nround_mean = 0
wloss_list = []
y_preds_exgal = []
for i in range(2):
    gc.collect()
    param['seed'] = np.random.randint(9999)
    ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                         feval=utils.lgb_multi_weighted_logloss_exgal,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    y_pred = ex.eval_oob(X_exgal, y_exgal, models, SEED, stratified=True, shuffle=True, 
                         n_class=y_exgal.unique().shape[0])
    y_preds_exgal.append(y_pred)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
    wloss_list.append( ret['wloss-mean'][-1] )

nround_mean = int((nround_mean/2) * 1.3)

result = f"CV wloss: {np.mean(wloss_list)} + {np.std(wloss_list)}"
print(result)

# =============================================================================
# model(exgal)
# =============================================================================

gc.collect()


np.random.seed(SEED)

models_exgal = []
for i in range(LOOP):
    print('building', i)
    gc.collect()
    param['seed'] = np.random.randint(9999)
    model = lgb.train(param, dtrain, num_boost_round=nround_mean, valid_sets=None, 
                      valid_names=None, fobj=None, feval=None, init_model=None, 
                      feature_name='auto', categorical_feature='auto', 
                      early_stopping_rounds=None, evals_result=None, 
                      verbose_eval=True, learning_rates=None, 
                      keep_training_booster=False, callbacks=None)
    
    models_exgal.append(model)


del dtrain, model_all; gc.collect()


# =============================================================================
# concat
# =============================================================================
for i,y_pred in enumerate(y_preds_gal):
    if i==0:
        y_pred_gal = y_pred
    else:
        y_pred_gal += y_pred
y_pred_gal /= len(y_preds_gal)
y_pred_gal = pd.DataFrame(y_pred_gal)
y_pred_gal.columns = [f'class_{di_gal[x]}' for x in range(len(di_gal))]

for i,y_pred in enumerate(y_preds_exgal):
    if i==0:
        y_pred_exgal = y_pred
    else:
        y_pred_exgal += y_pred
y_pred_exgal /= len(y_preds_exgal)
y_pred_exgal = pd.DataFrame(y_pred_exgal)
y_pred_exgal.columns = [f'class_{di_exgal[x]}' for x in range(len(di_exgal))]

y_pred = pd.concat([y_pred_gal, y_pred_exgal], 
                   ignore_index=True).fillna(0)

y_pred = y_pred[[f'class_{c}' for c in utils.classes]]



y = pd.concat([y_gal.replace(di_gal), y_exgal.replace(di_exgal)], 
              ignore_index=True)




loss = utils.multi_weighted_logloss(y.values, y_pred.values)

print(f'CV wloss: {loss}')

# =============================================================================
# weight
# =============================================================================
import utils_post

y_true = pd.get_dummies(y)

weight = utils_post.get_weight(y_true, y_pred.values, eta=0.1, nround=9999)
weight = np.append(weight, 1)
print(list(weight))

weight = np.array([0.87608653, 0.72238794, 0.2590674 , 0.1300367 , 1.11833844,
                   1.95931595, 0.3388923 , 2.29716221, 0.19067337, 0.90162932,
                   0.46783924, 0.06632756, 0.53624486, 0.86411252])

y_pred_tr = y_pred.copy()
# =============================================================================
# test
# =============================================================================
is_gal = pd.read_pickle('../data/te_is_gal.pkl')

files_te = sorted(glob('../data/test_f*.pkl'))

def read(f):
    df = pd.read_pickle(f)
    col_gal = list( set(df.columns) & set(COL_gal) )
    col_exgal = list( set(df.columns) & set(COL_exgal) )
    return df[is_gal][col_gal], df[~is_gal][col_exgal]

test_list = [read(f) for f in tqdm(files_te, mininterval=60)]


X_test_gal   = pd.concat([x[0] for x in test_list], axis=1)[COL_gal]
X_test_exgal = pd.concat([x[1] for x in test_list], axis=1)[COL_exgal]

del test_list; gc.collect()

print('predict gal')
for i,model in enumerate(tqdm(models_gal)):
    gc.collect()
    y_pred_gal = model.predict(X_test_gal)
    if i==0:
        y_pred_gal_all = y_pred_gal
    else:
        y_pred_gal_all += y_pred_gal

y_pred_gal_all /= len(models_gal)
y_pred_gal_all = pd.DataFrame(y_pred_gal_all)
y_pred_gal_all.columns = [f'class_{di_gal[x]}' for x in range(len(di_gal))]


print('predict exgal')
for i,model in enumerate(tqdm(models_exgal)):
    gc.collect()
    y_pred_exgal = model.predict(X_test_exgal)
    if i==0:
        y_pred_exgal_all = y_pred_exgal
    else:
        y_pred_exgal_all += y_pred_exgal

y_pred_exgal_all /= len(models_exgal)
y_pred_exgal_all = pd.DataFrame(y_pred_exgal_all)
y_pred_exgal_all.columns = [f'class_{di_exgal[x]}' for x in range(len(di_exgal))]



y_pred_all = pd.concat([y_pred_gal_all, y_pred_exgal_all], 
                       ignore_index=True).fillna(0)

y_pred_all = y_pred_all[[f'class_{c}' for c in utils.classes]]



oid = pd.concat([pd.read_pickle('../data/te_oid_gal.pkl'), 
                 pd.read_pickle('../data/te_oid_exgal.pkl')], 
            ignore_index=True)


sub = y_pred_all.copy()
sub['object_id'] = oid['object_id'].values
sub = sub[['object_id']+[f'class_{c}' for c in utils.classes]]


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

"""

SUBMIT_FILE_PATH = '../output/1110-1_post.csv.gz'

COMMENT = 'cur weight'

sub = pd.read_pickle('../data/y_pred_raw.pkl')
weight = np.array([0.8315192871685131, 0.9236110790771515, 0.20787756675658042, 
0.15388583406725131, 1.3287023446613135, 1.3542253901062942, 0.39899868175571807, 
2.438122807557637, 0.20053317780765934, 1.0631339564849398, 0.5212355765638879, 
0.07375455659250263, 0.5213834585484625, 0.9650715511377852, 1.0])

utils.postprocess(sub, weight, method='oli')

print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))

In [5]: print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))
   ...: 
class_99    0.575663
class_64    0.134502
class_92    0.052943
class_42    0.052751
class_15    0.041038
class_52    0.029075
class_16    0.028046
class_65    0.026185
class_88    0.022181
class_90    0.012165
class_67    0.011078
class_62    0.010848
class_95    0.002213
class_6     0.000929
class_53    0.000384
dtype: float64

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

utils.submit(SUBMIT_FILE_PATH, COMMENT)

"""

#==============================================================================
utils.end(__file__)
utils.stop_instance()



