#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:32:03 2018

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

SUBMIT_FILE_PATH = '../output/1113-1.csv.gz'

COMMENT = 'top400 features'

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

N_FEATURES = 400

# =============================================================================
# load
# =============================================================================
COL = pd.read_csv('LOG/imp_801_cv.py.csv').head(N_FEATURES).feature.tolist()

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
                         feval=utils.lgb_multi_weighted_logloss,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    y_pred = ex.eval_oob(X, y, models, SEED, stratified=True, shuffle=True, 
                         n_class=y.unique().shape[0])
    y_preds.append(y_pred)
    model_all += models
    nround_mean += len(ret['multi_logloss-mean'])
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

gc.collect()


np.random.seed(SEED)

model_all = []
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
    
    model_all.append(model)


del dtrain, X; gc.collect()

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


"""

SUBMIT_FILE_PATH = '../output/1111-1_post.csv.gz'

COMMENT = 'cur weight'

sub = pd.read_pickle('../data/y_pred_raw.pkl')
print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))

==== before: 1.283 ====
class_42    0.349992
class_90    0.216656
class_65    0.164483
class_62    0.052752
class_92    0.052643
class_15    0.045008
class_16    0.035043
class_88    0.022590
class_64    0.022131
class_52    0.019255
class_67    0.012529
class_6     0.004383
class_95    0.002175
class_53    0.000361
dtype: float64


weight = np.array([0.6982877770877463, 0.9570686160138555, 0.17924931579021885, 
0.1698051470477293, 1.6798908593635689, 1.1705951264354688, 0.47562338461914955, 
2.2612328100163808, 0.19469582519573891, 1.2710014023290797, 0.4606230830965776, 
0.06841068945315132, 0.4540140674600115, 0.946940888866237, 1.0])

utils.postprocess(sub, weight, method='oli')

print(sub.iloc[:, 1:].idxmax(1).value_counts(normalize=True))

==== after ====
class_99    0.673234
class_92    0.051506
class_52    0.048801
class_64    0.043668
class_42    0.037416
class_15    0.032857
class_16    0.027810
class_65    0.026261
class_88    0.016774
class_62    0.012978
class_67    0.012454
class_90    0.011902
class_95    0.002858
class_6     0.001115
class_53    0.000368







sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

utils.submit(SUBMIT_FILE_PATH, COMMENT)

"""
#==============================================================================
utils.end(__file__)
utils.stop_instance()
