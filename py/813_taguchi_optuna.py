#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:14:03 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import os, gc
from glob import glob
from tqdm import tqdm

import optuna

import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count


import utils, utils_metric
utils.start(__file__)
#==============================================================================

SEED = np.random.randint(9999)
print('SEED:', SEED)


NFOLD = 5

LOOP = 2

param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
#         'learning_rate': 0.01,
         'max_depth': 3,
#         'num_leaves': 63,
         'max_bin': 127,
         
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

taguchi_param = {
                'objective': 'multiclass',
                'num_class': 14,
                'nthread': cpu_count(),
                'learning_rate': 0.4,
                'max_depth': 3,
                'subsample': .9,
                'colsample_bytree': .7,
                'reg_alpha': .01,
                'reg_lambda': .01,
                'min_split_gain': 0.01,
                'min_child_weight': 200,
                'verbose': -1,
                
                'max_bin': 20,
                'bagging_freq': 1,
            }

param = taguchi_param

DROP = ['0_minus_1_q2575_rng',
        '1_minus_2_q2575_rng',
        '2_minus_3_q2575_rng',
        '3_minus_4_q2575_rng',
        '4_minus_5_q2575_rng',
        '5_minus_0_q2575_rng',
        'band-0_std_upper_flux_quantile10',
        'band-1_std_upper_flux_quantile10',
        'band-2_std_upper_flux_quantile10',
        'band-3_std_upper_flux_quantile10',
        'band-4_std_upper_flux_quantile10',
        'band-5_std_upper_flux_quantile10',
        'band-0_std_upper_flux_quantile90',
        'band-1_std_upper_flux_quantile90',
        'band-2_std_upper_flux_quantile90',
        'band-3_std_upper_flux_quantile90',
        'band-4_std_upper_flux_quantile90',
        'band-5_std_upper_flux_quantile90',
        '0_minus_1_max',
        '1_minus_2_max',
        '2_minus_3_max',
        '3_minus_4_max',
        '4_minus_5_max',
        '5_minus_0_max',
        'abs_magnitude_mean',
        'abs_magnitude_median',
        'abs_magnitude_std',
        'abs_magnitude_var',
        'abs_magnitude_skew',
        '0_minus_1_ratsqmax',
        '1_minus_2_ratsqmax',
        '2_minus_3_ratsqmax',
        '3_minus_4_ratsqmax',
        '4_minus_5_ratsqmax',
        '5_minus_0_ratsqmax']


# =============================================================================
# load
# =============================================================================


X = pd.read_pickle('../FROM_MYTEAM/guchio_xtrain_20181208.pkl.gz')
y = utils.load_target().target

X.drop(DROP+['target', 'object_id'], axis=1, inplace=True)

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
# cv1
# =============================================================================

dtrain = lgb.Dataset(X, y, free_raw_data=False)
gc.collect()

def objective(trial):
    param['max_depth'] = trial.suggest_int('max_depth', 3, 6)
    param['subsample'] = trial.suggest_discrete_uniform('subsample', 0.5, 1.0, 0.1)
    param['colsample_bytree'] = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1.0, 0.1)
    param['reg_alpha'] = trial.suggest_discrete_uniform('reg_alpha', 0.01, 0.5, 0.01)
    param['reg_lambda'] = trial.suggest_discrete_uniform('reg_lambda', 0.01, 0.5, 0.01)
    param['min_split_gain'] = trial.suggest_discrete_uniform('min_split_gain', 0.01, 0.1, 0.01)
    param['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 300)
#    param['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 300)
    param['max_bin'] = trial.suggest_int('max_bin', 15, 127)
    
    model_all = []
    nround_mean = 0
    wloss_list = []
    for i in range(LOOP):
        gc.collect()
        param['seed'] = np.random.randint(9999)
        ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD, 
                             fobj=utils_metric.wloss_objective, 
                             feval=utils_metric.wloss_metric,
                             early_stopping_rounds=100, verbose_eval=100,
                             seed=SEED+i)
        model_all += models
        nround_mean += len(ret['multi_logloss-mean'])
        wloss_list.append( ret['wloss-mean'][-1] )
    
    return np.mean(wloss_list)


# =============================================================================
# optuna
# =============================================================================

# optuna
study = optuna.create_study()
study.optimize(objective, n_trials=9999)

# 最適解
print(study.best_params)
print(study.best_value)
print(study.best_trial)

"""
0.47523778676986694
FrozenTrial(trial_id=87, state=<TrialState.COMPLETE: 1>, value=0.47523778676986694, datetime_start=datetime.datetime(2018, 12, 6, 0, 9, 7, 785038), datetime_complete=datetime.datetime(2018, 12, 6, 0, 12, 19, 561225), params={'f014_detected_dates': 1, 'f001_hostgal_photoz_err': 1, 'f002_flux_by_flux_ratio_sq_skew': 1, 'f002_lumi_std-d-mean': 1, 'f014_d1_lumi_min': 0, 'f013_lumi_q10': 1, 'f002_lumi_kurt': 1, 'f013_lumi_min': 0, 'f017_pb0_lumi_mean-d-pb5_lumi_mean': 0, 'f010_pb5_flux_norm2_min': 0, 'f001_hostgal_photoz': 1, 'f003_pb2_flux_err_q25-d-pb3_flux_err_q25': 1, 'f002_lumi_max-m-min-d-mean': 0, 'f003_pb0_lumi_std-d-pb4_lumi_std': 1, 'f003_pb1_flux_q75-m-q25': 1, 'f014_d1_lumi_max-d-min': 1, 'f014_d1_flux_norm3_min': 0, 'f002_flux_ratio_sq_max-m-min-d-mean': 0, 'f003_pb0_flux_norm2_q25-d-pb1_flux_norm2_q25': 0, 'f003_pb2_lumi_q10-d-pb5_lumi_q10': 1, 'f015_pb5_d1_lumi_q25': 0, 'f017_pb0_lumi_median-d-pb5_lumi_median': 1, 'f014_d0_flux_ratio_sq_median': 0, 'f003_pb2_flux_err_median-d-pb3_flux_err_median': 0, 'f014_d0_flux_ratio_sq_q75': 1, 'f003_pb0_lumi_max-d-pb4_lumi_max': 0, 'f003_pb5_flux_norm2_min': 0, 'f015_pb2_d1_flux_norm3_min': 1, 'f002_flux_by_flux_ratio_sq_std-d-mean': 1, 'f014_d1_flux_norm1_min': 1, 'f010_pb5_flux_q75-m-q25': 0, 'f003_pb2_flux_min-d-pb5_flux_min': 1, 'f013_lumi_median': 1, 'f015_pb1_d0_flux_ratio_sq_median': 1, 'f015_pb5_d1_flux_norm1_max': 1, 'f013_flux_ratio_sq_std-d-mean': 1, 'f003_pb1_lumi_q90-d-pb2_lumi_q90': 1, 'f003_pb1_lumi_max-m-min': 1, 'f003_pb0_lumi_max': 0, 'f002_lumi_mean': 0, 'f014_d1_flux_norm3_q90': 0, 'f014_d1_flux_norm3_q75': 1, 'f003_pb1_lumi_q75-m-q25': 0, 'f010_pb0_flux_ratio_sq_q75-d-pb4_flux_ratio_sq_q75': 1, 'f015_pb4_d1_flux_norm3_max-d-min': 0, 'f015_pb1_d0_lumi_max': 0, 'f013_lumi_max-d-min': 1, 'f013_lumi_std-d-mean': 0, 'f015_pb5_d1_flux_max-d-min': 1, 'f014_d1_lumi_max': 1, 'f014_d0_flux_ratio_sq_q75-m-q25': 0, 'f002_a_2__skewness': 0, 'f003_pb2_flux_norm1_min-d-pb5_flux_norm1_min': 0, 'f003_pb4_lumi_mean': 0, 'f003_pb1_lumi_max-d-min-d-pb3_lumi_max-d-min': 1, 'f003_pb0_lumi_max-d-pb5_lumi_max': 1, 'f003_pb0_flux_ratio_sq_mean-d-pb4_flux_ratio_sq_mean': 1, 'f017_pb0_lumi_mean-d-pb4_lumi_mean': 0, 'f015_pb5_d1_flux_mean': 0, 'f015_pb5_d1_lumi_q75': 1, 'f003_pb0_lumi_mean-d-pb3_lumi_mean': 0, 'f003_pb0_flux_by_flux_ratio_sq_max-d-pb5_flux_by_flux_ratio_sq_max': 1, 'f003_pb5_lumi_mean': 0, 'f003_pb0_lumi_mean-d-pb2_lumi_mean': 0, 'f003_pb1_lumi_q90-m-q10': 0, 'f017_pb3_lumi_std-d-mean': 1, 'f015_pb3_d1_lumi_max-d-pb5_d1_lumi_max': 1, 'f003_pb2_lumi_q90': 1, 'f015_pb3_d1_flux_norm1_max-d-pb5_d1_flux_norm1_max': 1, 'f003_pb2_flux_q75-m-q25': 1, 'f003_pb0_flux_by_flux_ratio_sq_q75-m-q25-d-pb4_flux_by_flux_ratio_sq_q75-m-q25': 1, 'f015_pb2_d1_flux_by_flux_ratio_sq_q25-d-pb4_d1_flux_by_flux_ratio_sq_q25': 0, 'f015_pb2_d1_lumi_min': 0, 'f003_pb3_flux_norm3_skew': 0, 'f003_pb1_flux_norm1_q75': 1, 'f003_pb4_lumi_std-d-mean': 1, 'f003_pb1_lumi_mean-d-pb3_lumi_mean': 1, 'f023_pb4_lumi_min': 0, 'f003_pb0_flux_min-d-pb2_flux_min': 0, 'f015_pb1_d0_lumi_std': 1, 'f003_pb1_flux_ratio_sq_median': 0, 'f017_pb0_lumi_q75-d-pb5_lumi_q75': 1, 'f003_pb0_flux_by_flux_ratio_sq_q90-d-pb4_flux_by_flux_ratio_sq_q90': 0, 'f003_pb1_lumi_q75-d-pb5_lumi_q75': 1, 'f017_pb2_flux_by_flux_ratio_sq_median': 1, 'f014_d1_flux_norm3_median': 0, 'f003_pb2_flux_norm1_q90-m-q10': 1, 'f003_pb0_flux_by_flux_ratio_sq_q90-d-pb5_flux_by_flux_ratio_sq_q90': 1, 'f004_y0_detected_std-d-mean-d-y2_detected_std-d-mean': 1, 'f014_d1_lumi_std': 1, 'f003_pb3_lumi_max-d-pb4_lumi_max': 0, 'f002_flux_by_flux_ratio_sq_max-m-min-d-mean': 1, 'f003_pb1_lumi_std': 1, 'f003_pb0_lumi_max-d-min-d-pb5_lumi_max-d-min': 0, 'f003_pb0_flux_by_flux_ratio_sq_min-d-pb2_flux_by_flux_ratio_sq_min': 1, 'f003_pb1_lumi_q75-m-q25-d-pb5_lumi_q75-m-q25': 1, 'f003_pb1_flux_norm1_q75-m-q25': 0, 'f017_pb4_lumi_std-d-mean': 0, 'f014_d1_lumi_max-m-min': 0, 'f003_pb1_lumi_q10-d-pb5_lumi_q10': 0, 'f014_d1_lumi_std-d-mean': 1, 'f014_d0_lumi_std-d-mean': 0, 'fbk1-004_pb1_flux_q25': 0, 'f003_pb2_lumi_kurt': 1, 'f003_pb0_flux_ratio_sq_mean-d-pb5_flux_ratio_sq_mean': 1, 'f004_y2_flux_ratio_sq_skew': 1, 'f003_pb0_flux_norm2_max-d-min': 0, 'fbk1-007_pb1_flux_std': 1, 'f003_pb0_lumi_q75-d-pb3_lumi_q75': 1, 'f017_pb2_flux_norm2_q75': 1, 'f003_pb2_flux_err_min-d-pb4_flux_err_min': 1, 'f014_d1_flux_norm3_q25': 0, 'f017_pb3_lumi_max-d-min': 0, 'f027_lumi_skew': 1, 'f003_pb0_lumi_std-d-pb3_lumi_std': 0, 'f014_d1_flux_norm3_q10': 1, 'f003_pb0_lumi_q90-d-pb2_lumi_q90': 0, 'f003_pb1_lumi_q75': 0, 'f024_pb5_flux_norm2_median': 0, 'f003_pb0_flux_ratio_sq_q75-d-pb3_flux_ratio_sq_q75': 1, 'f010_pb5_flux_norm3_q75-m-q25': 1, 'f003_pb1_lumi_std-d-pb3_lumi_std': 1, 'f015_pb5_d1_flux_norm2_max-d-min': 0, 'f014_d1_flux_norm3_max-d-min': 1, 'fbk1-007_pb1_flux_max': 1, 'f023_pb4_lumi_q10': 1, 'f003_pb1_lumi_q90': 0, 'f015_pb2_d0_lumi_median': 0, 'f015_pb2_d1_flux_min-d-pb5_d1_flux_min': 1, 'f017_pb3_lumi_min': 0, 'f015_pb5_d1_flux_norm1_q90': 0, 'f015_pb2_d1_flux_max-m-min-d-pb5_d1_flux_max-m-min': 1, 'f015_pb2_d0_flux_ratio_sq_median': 1, 'f015_pb2_d0_flux_ratio_sq_q75': 1, 'f014_d0_flux_ratio_sq_q25': 1, 'f003_pb0_flux_ratio_sq_q10-d-pb3_flux_ratio_sq_q10': 1, 'f023_pb4_flux_norm1_median': 1, 'f020_pb4_lumi_q25-d-pb5_lumi_q25': 1, 'f015_pb2_detected_dates_max-m-min': 0, 'f017_pb4_flux_norm2_q75': 1, 'f002_lumi_skew': 0, 'f002_flux_norm1_mean': 0, 'f013_flux_by_flux_ratio_sq_std-d-mean': 0, 'f003_pb5_flux_q75-m-q25': 0, 'f015_pb1_d0_lumi_q25-d-pb2_d0_lumi_q25': 0, 'f002_a_5__fft_coefficient__coeff_1__attr_"abs"': 0, 'f003_pb2_flux_q90-m-q10': 1, 'f015_pb5_d1_flux_norm1_max-d-min': 1, 'f003_pb1_lumi_max-d-pb4_lumi_max': 1, 'f005_pb2_y1_flux_q75-m-q25': 0, 'f003_pb3_lumi_std-d-pb5_lumi_std': 1, 'f003_pb1_lumi_q75-d-pb2_lumi_q75': 1, 'f003_pb2_lumi_max-d-pb5_lumi_max': 0, 'f015_pb0_d1_flux_norm2_min-d-pb5_d1_flux_norm2_min': 0, 'f014_d0_flux_by_flux_ratio_sq_q25': 1, 'f017_pb3_lumi_q10': 1, 'f003_pb3_lumi_std-d-mean': 1, 'f003_pb5_flux_std': 0, 'f014_d1_flux_min': 0, 'f015_pb0_d1_flux_norm2_max-d-pb2_d1_flux_norm2_max': 0, 'f017_pb5_lumi_min': 1, 'f023_pb1_flux_norm1_mean': 1, 'f015_pb4_d1_lumi_std': 1, 'f015_pb3_d0_flux_ratio_sq_median': 1, 'f015_pb2_d0_lumi_kurt-d-pb3_d0_lumi_kurt': 1, 'f003_pb2_lumi_std-d-mean': 0, 'f014_d1_flux_std': 1, 'f003_pb2_flux_skew': 1, 'f003_pb2_lumi_max-d-pb3_lumi_max': 1, 'f024_pb5_flux_norm2_mean': 0, 'f003_pb0_flux_norm2_q10-d-pb5_flux_norm2_q10': 0, 'f004_y1_lumi_kurt': 0, 'f005_pb0_y2_lumi_std-d-mean': 1, 'f014_d1_flux_norm3_mean': 1, 'f014_d1_flux_by_flux_ratio_sq_max-d-min': 0, 'f015_pb1_d1_lumi_min': 1, 'f026_pb2_date_diff_0.3': 1, 'f015_pb3_d0_lumi_skew-d-pb4_d0_lumi_skew': 0, 'f002_flux_by_flux_ratio_sq_max-d-min': 1, 'f003_pb3_lumi_q75-d-pb5_lumi_q75': 1, 'f002_flux_skew': 1, 'f003_pb2_flux_err_q25-d-pb4_flux_err_q25': 0, 'f003_pb0_flux_ratio_sq_q25-d-pb4_flux_ratio_sq_q25': 1, 'f003_pb2_lumi_median-d-pb4_lumi_median': 0, 'f003_pb0_lumi_max-d-pb3_lumi_max': 1, 'f010_pb2_lumi_skew-d-pb3_lumi_skew': 0, 'f003_pb2_lumi_q75': 0, 'f015_pb1_d0_flux_ratio_sq_mean': 0, 'f014_d1_lumi_q75-m-q25': 1, 'f020_pb5_lumi_mean': 0, 'f017_pb2_flux_norm2_q75-d-pb5_flux_norm2_q75': 1, 'f015_pb1_d1_flux_norm1_max-m-min': 0, 'f003_pb1_lumi_mean': 1, 'f002_flux_ratio_sq_median': 0, 'f023_pb4_flux_norm2_q75': 0, 'f004_y2_flux_ratio_sq_kurt': 0, 'f005_pb1_y0_lumi_std-d-mean': 1, 'f003_pb2_lumi_max-d-min': 1, 'f014_d0_flux_norm2_min': 0, 'f015_pb5_d0_flux_err_skew': 1, 'f003_pb1_lumi_max-d-min-d-pb4_lumi_max-d-min': 1, 'f017_pb5_flux_norm1_mean': 0, 'f027_lumi_std': 1, 'f002_flux_ratio_sq_q90-m-q10': 0, 'f004_y1_lumi_median-d-y2_lumi_median': 1, 'f015_pb5_d1_lumi_q90': 0, 'f003_pb0_lumi_max-m-min-d-pb3_lumi_max-m-min': 1, 'f003_pb2_lumi_mean-d-pb3_lumi_mean': 1, 'f010_pb0_flux_err_mean-d-pb4_flux_err_mean': 1, 'f003_pb2_lumi_median': 0, 'f015_pb0_d0_flux_norm3_q90-m-q10-d-pb5_d0_flux_norm3_q90-m-q10': 1, 'f003_pb2_lumi_q75-d-pb5_lumi_q75': 1, 'f003_pb2_flux_norm1_min-d-pb3_flux_norm1_min': 0, 'f003_pb2_flux_norm2_q75-d-pb4_flux_norm2_q75': 0, 'f003_pb0_flux_norm1_max-d-pb3_flux_norm1_max': 0, 'f015_pb1_d0_lumi_max-m-min': 1, 'f003_pb1_lumi_q75-m-q25-d-pb3_lumi_q75-m-q25': 1, 'f003_pb1_lumi_q90-m-q10-d-pb2_lumi_q90-m-q10': 1, 'f027_lumi_q25': 0, 'f003_pb1_lumi_max': 1, 'fbk1-007_pb0_flux_std': 1, 'f023_pb5_flux_norm3_q75': 0, 'f027_lumi_kurt': 0, 'f003_pb2_flux_norm1_skew': 0, 'f017_pb1_lumi_q90-d-pb3_lumi_q90': 1, 'f003_pb0_lumi_max-m-min-d-pb5_lumi_max-m-min': 0, 'f025_pb2_flux_by_flux_ratio_sq_min': 1, 'f010_pb4_lumi_median-d-pb5_lumi_median': 0, 'fbk1-006_pb1_flux_q75': 0, 'f005_pb1_y2_lumi_max-d-min': 0, 'f023_pb3_lumi_mean-d-pb5_lumi_mean': 1, 'f003_pb2_lumi_q75-d-pb4_lumi_q75': 1, 'f003_pb0_lumi_max-m-min': 0, 'f015_pb1_d1_flux_norm1_max-d-pb3_d1_flux_norm1_max': 1, 'f017_pb2_lumi_q75-d-pb4_lumi_q75': 0, 'f023_pb5_flux_norm1_median': 1, 'f003_pb1_lumi_skew-d-pb3_lumi_skew': 1, 'f024_pb5_flux_q10': 0, 'f003_pb0_lumi_std-d-pb5_lumi_std': 1, 'f014_d1_lumi_q90-m-q10': 0, 'f002_flux_by_flux_ratio_sq_median': 1, 'f025_pb2_lumi_q10-d-pb3_lumi_q10': 1, 'f003_pb0_lumi_median-d-pb1_lumi_median': 1, 'f003_pb1_lumi_q75-d-pb4_lumi_q75': 1, 'f015_pb2_d0_flux_err_q25-d-pb5_d0_flux_err_q25': 1, 'f010_pb0_flux_ratio_sq_median-d-pb4_flux_ratio_sq_median': 1, 'f023_pb3_lumi_q75-d-pb5_lumi_q75': 0, 'f017_pb4_lumi_max-d-min': 1, 'f005_pb5_y2_lumi_median': 0, 'f013_lumi_q90-m-q10': 0, 'f003_pb1_flux_err_min-d-pb3_flux_err_min': 0, 'f023_pb2_flux_by_flux_ratio_sq_median': 0, 'f003_pb4_flux_kurt': 1, 'f014_d1_flux_norm1_std': 1, 'f023_pb4_lumi_q25-d-pb5_lumi_q25': 0, 'f012_pb3_lumi_min-d-pb4_lumi_min': 1, 'f003_pb2_flux_ratio_sq_median-d-pb5_flux_ratio_sq_median': 1, 'f015_pb3_d1_flux_max-d-pb5_d1_flux_max': 0, 'f017_pb5_flux_norm1_q90': 1, 'f017_pb2_lumi_q25-d-pb4_lumi_q25': 0, 'f017_pb1_flux_err_max': 1, 'f003_pb1_lumi_max-d-pb5_lumi_max': 0, 'f023_pb2_lumi_std': 1, 'f003_pb0_flux_norm2_std-d-mean-d-pb4_flux_norm2_std-d-mean': 1, 'f015_pb2_d1_lumi_max-d-pb5_d1_lumi_max': 1, 'f012_pb2_flux_by_flux_ratio_sq_min': 1, 'f003_pb2_lumi_q75-m-q25-d-pb3_lumi_q75-m-q25': 0, 'f005_pb0_y2_flux_norm2_std-d-mean': 0, 'f003_pb0_flux_ratio_sq_q90-d-pb5_flux_ratio_sq_q90': 0, 'f013_lumi_std': 0, 'f003_pb0_lumi_skew': 1, 'f018_pb1_lumi_median-d-pb2_lumi_median': 0, 'f020_pb4_lumi_min': 1, 'f026_pb3_date_diff_0.3': 0, 'f005_pb5_y2_flux_norm2_std-d-mean': 0, 'f003_pb2_lumi_mean-d-pb4_lumi_mean': 1, 'f014_d0_flux_ratio_sq_q90': 0, 'f015_pb0_d0_lumi_q90-d-pb2_d0_lumi_q90': 1, 'f013_lumi_q75-m-q25': 0, 'f003_pb1_lumi_max-d-pb3_lumi_max': 0, 'f004_y0_lumi_q25-d-y1_lumi_q25': 0, 'f002_flux_norm1_q90': 0, 'f003_pb0_flux_norm1_median': 0, 'fbk1-006_pb0_flux_mean': 1, 'f015_pb5_d0_flux_norm3_std-d-mean': 0, 'f012_pb5_flux_norm2_mean': 0, 'f003_pb0_flux_ratio_sq_q75-d-pb4_flux_ratio_sq_q75': 1, 'f018_pb1_lumi_median-d-pb4_lumi_median': 1, 'f017_pb2_flux_norm2_mean-d-pb4_flux_norm2_mean': 1, 'f015_pb1_d1_flux_by_flux_ratio_sq_max-d-min': 0, 'f023_pb5_flux_mean': 1, 'f003_pb1_lumi_q75-d-pb3_lumi_q75': 0, 'f003_pb0_lumi_std-d-mean-d-pb1_lumi_std-d-mean': 1, 'f015_pb0_d0_lumi_kurt-d-pb1_d0_lumi_kurt': 1, 'f003_pb0_lumi_mean-d-pb1_lumi_mean': 1, 'f023_pb4_flux_norm3_q90': 1, 'f017_pb3_flux_norm1_min': 1, 'f003_pb2_lumi_max-d-pb4_lumi_max': 0, 'f010_pb4_lumi_std-d-mean-d-pb5_lumi_std-d-mean': 0, 'f010_pb2_flux_err_median-d-pb3_flux_err_median': 1}, user_attrs={}, system_attrs={}, intermediate_values={}, params_in_internal_repr={'f014_detected_dates': 1, 'f001_hostgal_photoz_err': 1, 'f002_flux_by_flux_ratio_sq_skew': 1, 'f002_lumi_std-d-mean': 1, 'f014_d1_lumi_min': 0, 'f013_lumi_q10': 1, 'f002_lumi_kurt': 1, 'f013_lumi_min': 0, 'f017_pb0_lumi_mean-d-pb5_lumi_mean': 0, 'f010_pb5_flux_norm2_min': 0, 'f001_hostgal_photoz': 1, 'f003_pb2_flux_err_q25-d-pb3_flux_err_q25': 1, 'f002_lumi_max-m-min-d-mean': 0, 'f003_pb0_lumi_std-d-pb4_lumi_std': 1, 'f003_pb1_flux_q75-m-q25': 1, 'f014_d1_lumi_max-d-min': 1, 'f014_d1_flux_norm3_min': 0, 'f002_flux_ratio_sq_max-m-min-d-mean': 0, 'f003_pb0_flux_norm2_q25-d-pb1_flux_norm2_q25': 0, 'f003_pb2_lumi_q10-d-pb5_lumi_q10': 1, 'f015_pb5_d1_lumi_q25': 0, 'f017_pb0_lumi_median-d-pb5_lumi_median': 1, 'f014_d0_flux_ratio_sq_median': 0, 'f003_pb2_flux_err_median-d-pb3_flux_err_median': 0, 'f014_d0_flux_ratio_sq_q75': 1, 'f003_pb0_lumi_max-d-pb4_lumi_max': 0, 'f003_pb5_flux_norm2_min': 0, 'f015_pb2_d1_flux_norm3_min': 1, 'f002_flux_by_flux_ratio_sq_std-d-mean': 1, 'f014_d1_flux_norm1_min': 1, 'f010_pb5_flux_q75-m-q25': 0, 'f003_pb2_flux_min-d-pb5_flux_min': 1, 'f013_lumi_median': 1, 'f015_pb1_d0_flux_ratio_sq_median': 1, 'f015_pb5_d1_flux_norm1_max': 1, 'f013_flux_ratio_sq_std-d-mean': 1, 'f003_pb1_lumi_q90-d-pb2_lumi_q90': 1, 'f003_pb1_lumi_max-m-min': 1, 'f003_pb0_lumi_max': 0, 'f002_lumi_mean': 0, 'f014_d1_flux_norm3_q90': 0, 'f014_d1_flux_norm3_q75': 1, 'f003_pb1_lumi_q75-m-q25': 0, 'f010_pb0_flux_ratio_sq_q75-d-pb4_flux_ratio_sq_q75': 1, 'f015_pb4_d1_flux_norm3_max-d-min': 0, 'f015_pb1_d0_lumi_max': 0, 'f013_lumi_max-d-min': 1, 'f013_lumi_std-d-mean': 0, 'f015_pb5_d1_flux_max-d-min': 1, 'f014_d1_lumi_max': 1, 'f014_d0_flux_ratio_sq_q75-m-q25': 0, 'f002_a_2__skewness': 0, 'f003_pb2_flux_norm1_min-d-pb5_flux_norm1_min': 0, 'f003_pb4_lumi_mean': 0, 'f003_pb1_lumi_max-d-min-d-pb3_lumi_max-d-min': 1, 'f003_pb0_lumi_max-d-pb5_lumi_max': 1, 'f003_pb0_flux_ratio_sq_mean-d-pb4_flux_ratio_sq_mean': 1, 'f017_pb0_lumi_mean-d-pb4_lumi_mean': 0, 'f015_pb5_d1_flux_mean': 0, 'f015_pb5_d1_lumi_q75': 1, 'f003_pb0_lumi_mean-d-pb3_lumi_mean': 0, 'f003_pb0_flux_by_flux_ratio_sq_max-d-pb5_flux_by_flux_ratio_sq_max': 1, 'f003_pb5_lumi_mean': 0, 'f003_pb0_lumi_mean-d-pb2_lumi_mean': 0, 'f003_pb1_lumi_q90-m-q10': 0, 'f017_pb3_lumi_std-d-mean': 1, 'f015_pb3_d1_lumi_max-d-pb5_d1_lumi_max': 1, 'f003_pb2_lumi_q90': 1, 'f015_pb3_d1_flux_norm1_max-d-pb5_d1_flux_norm1_max': 1, 'f003_pb2_flux_q75-m-q25': 1, 'f003_pb0_flux_by_flux_ratio_sq_q75-m-q25-d-pb4_flux_by_flux_ratio_sq_q75-m-q25': 1, 'f015_pb2_d1_flux_by_flux_ratio_sq_q25-d-pb4_d1_flux_by_flux_ratio_sq_q25': 0, 'f015_pb2_d1_lumi_min': 0, 'f003_pb3_flux_norm3_skew': 0, 'f003_pb1_flux_norm1_q75': 1, 'f003_pb4_lumi_std-d-mean': 1, 'f003_pb1_lumi_mean-d-pb3_lumi_mean': 1, 'f023_pb4_lumi_min': 0, 'f003_pb0_flux_min-d-pb2_flux_min': 0, 'f015_pb1_d0_lumi_std': 1, 'f003_pb1_flux_ratio_sq_median': 0, 'f017_pb0_lumi_q75-d-pb5_lumi_q75': 1, 'f003_pb0_flux_by_flux_ratio_sq_q90-d-pb4_flux_by_flux_ratio_sq_q90': 0, 'f003_pb1_lumi_q75-d-pb5_lumi_q75': 1, 'f017_pb2_flux_by_flux_ratio_sq_median': 1, 'f014_d1_flux_norm3_median': 0, 'f003_pb2_flux_norm1_q90-m-q10': 1, 'f003_pb0_flux_by_flux_ratio_sq_q90-d-pb5_flux_by_flux_ratio_sq_q90': 1, 'f004_y0_detected_std-d-mean-d-y2_detected_std-d-mean': 1, 'f014_d1_lumi_std': 1, 'f003_pb3_lumi_max-d-pb4_lumi_max': 0, 'f002_flux_by_flux_ratio_sq_max-m-min-d-mean': 1, 'f003_pb1_lumi_std': 1, 'f003_pb0_lumi_max-d-min-d-pb5_lumi_max-d-min': 0, 'f003_pb0_flux_by_flux_ratio_sq_min-d-pb2_flux_by_flux_ratio_sq_min': 1, 'f003_pb1_lumi_q75-m-q25-d-pb5_lumi_q75-m-q25': 1, 'f003_pb1_flux_norm1_q75-m-q25': 0, 'f017_pb4_lumi_std-d-mean': 0, 'f014_d1_lumi_max-m-min': 0, 'f003_pb1_lumi_q10-d-pb5_lumi_q10': 0, 'f014_d1_lumi_std-d-mean': 1, 'f014_d0_lumi_std-d-mean': 0, 'fbk1-004_pb1_flux_q25': 0, 'f003_pb2_lumi_kurt': 1, 'f003_pb0_flux_ratio_sq_mean-d-pb5_flux_ratio_sq_mean': 1, 'f004_y2_flux_ratio_sq_skew': 1, 'f003_pb0_flux_norm2_max-d-min': 0, 'fbk1-007_pb1_flux_std': 1, 'f003_pb0_lumi_q75-d-pb3_lumi_q75': 1, 'f017_pb2_flux_norm2_q75': 1, 'f003_pb2_flux_err_min-d-pb4_flux_err_min': 1, 'f014_d1_flux_norm3_q25': 0, 'f017_pb3_lumi_max-d-min': 0, 'f027_lumi_skew': 1, 'f003_pb0_lumi_std-d-pb3_lumi_std': 0, 'f014_d1_flux_norm3_q10': 1, 'f003_pb0_lumi_q90-d-pb2_lumi_q90': 0, 'f003_pb1_lumi_q75': 0, 'f024_pb5_flux_norm2_median': 0, 'f003_pb0_flux_ratio_sq_q75-d-pb3_flux_ratio_sq_q75': 1, 'f010_pb5_flux_norm3_q75-m-q25': 1, 'f003_pb1_lumi_std-d-pb3_lumi_std': 1, 'f015_pb5_d1_flux_norm2_max-d-min': 0, 'f014_d1_flux_norm3_max-d-min': 1, 'fbk1-007_pb1_flux_max': 1, 'f023_pb4_lumi_q10': 1, 'f003_pb1_lumi_q90': 0, 'f015_pb2_d0_lumi_median': 0, 'f015_pb2_d1_flux_min-d-pb5_d1_flux_min': 1, 'f017_pb3_lumi_min': 0, 'f015_pb5_d1_flux_norm1_q90': 0, 'f015_pb2_d1_flux_max-m-min-d-pb5_d1_flux_max-m-min': 1, 'f015_pb2_d0_flux_ratio_sq_median': 1, 'f015_pb2_d0_flux_ratio_sq_q75': 1, 'f014_d0_flux_ratio_sq_q25': 1, 'f003_pb0_flux_ratio_sq_q10-d-pb3_flux_ratio_sq_q10': 1, 'f023_pb4_flux_norm1_median': 1, 'f020_pb4_lumi_q25-d-pb5_lumi_q25': 1, 'f015_pb2_detected_dates_max-m-min': 0, 'f017_pb4_flux_norm2_q75': 1, 'f002_lumi_skew': 0, 'f002_flux_norm1_mean': 0, 'f013_flux_by_flux_ratio_sq_std-d-mean': 0, 'f003_pb5_flux_q75-m-q25': 0, 'f015_pb1_d0_lumi_q25-d-pb2_d0_lumi_q25': 0, 'f002_a_5__fft_coefficient__coeff_1__attr_"abs"': 0, 'f003_pb2_flux_q90-m-q10': 1, 'f015_pb5_d1_flux_norm1_max-d-min': 1, 'f003_pb1_lumi_max-d-pb4_lumi_max': 1, 'f005_pb2_y1_flux_q75-m-q25': 0, 'f003_pb3_lumi_std-d-pb5_lumi_std': 1, 'f003_pb1_lumi_q75-d-pb2_lumi_q75': 1, 'f003_pb2_lumi_max-d-pb5_lumi_max': 0, 'f015_pb0_d1_flux_norm2_min-d-pb5_d1_flux_norm2_min': 0, 'f014_d0_flux_by_flux_ratio_sq_q25': 1, 'f017_pb3_lumi_q10': 1, 'f003_pb3_lumi_std-d-mean': 1, 'f003_pb5_flux_std': 0, 'f014_d1_flux_min': 0, 'f015_pb0_d1_flux_norm2_max-d-pb2_d1_flux_norm2_max': 0, 'f017_pb5_lumi_min': 1, 'f023_pb1_flux_norm1_mean': 1, 'f015_pb4_d1_lumi_std': 1, 'f015_pb3_d0_flux_ratio_sq_median': 1, 'f015_pb2_d0_lumi_kurt-d-pb3_d0_lumi_kurt': 1, 'f003_pb2_lumi_std-d-mean': 0, 'f014_d1_flux_std': 1, 'f003_pb2_flux_skew': 1, 'f003_pb2_lumi_max-d-pb3_lumi_max': 1, 'f024_pb5_flux_norm2_mean': 0, 'f003_pb0_flux_norm2_q10-d-pb5_flux_norm2_q10': 0, 'f004_y1_lumi_kurt': 0, 'f005_pb0_y2_lumi_std-d-mean': 1, 'f014_d1_flux_norm3_mean': 1, 'f014_d1_flux_by_flux_ratio_sq_max-d-min': 0, 'f015_pb1_d1_lumi_min': 1, 'f026_pb2_date_diff_0.3': 1, 'f015_pb3_d0_lumi_skew-d-pb4_d0_lumi_skew': 0, 'f002_flux_by_flux_ratio_sq_max-d-min': 1, 'f003_pb3_lumi_q75-d-pb5_lumi_q75': 1, 'f002_flux_skew': 1, 'f003_pb2_flux_err_q25-d-pb4_flux_err_q25': 0, 'f003_pb0_flux_ratio_sq_q25-d-pb4_flux_ratio_sq_q25': 1, 'f003_pb2_lumi_median-d-pb4_lumi_median': 0, 'f003_pb0_lumi_max-d-pb3_lumi_max': 1, 'f010_pb2_lumi_skew-d-pb3_lumi_skew': 0, 'f003_pb2_lumi_q75': 0, 'f015_pb1_d0_flux_ratio_sq_mean': 0, 'f014_d1_lumi_q75-m-q25': 1, 'f020_pb5_lumi_mean': 0, 'f017_pb2_flux_norm2_q75-d-pb5_flux_norm2_q75': 1, 'f015_pb1_d1_flux_norm1_max-m-min': 0, 'f003_pb1_lumi_mean': 1, 'f002_flux_ratio_sq_median': 0, 'f023_pb4_flux_norm2_q75': 0, 'f004_y2_flux_ratio_sq_kurt': 0, 'f005_pb1_y0_lumi_std-d-mean': 1, 'f003_pb2_lumi_max-d-min': 1, 'f014_d0_flux_norm2_min': 0, 'f015_pb5_d0_flux_err_skew': 1, 'f003_pb1_lumi_max-d-min-d-pb4_lumi_max-d-min': 1, 'f017_pb5_flux_norm1_mean': 0, 'f027_lumi_std': 1, 'f002_flux_ratio_sq_q90-m-q10': 0, 'f004_y1_lumi_median-d-y2_lumi_median': 1, 'f015_pb5_d1_lumi_q90': 0, 'f003_pb0_lumi_max-m-min-d-pb3_lumi_max-m-min': 1, 'f003_pb2_lumi_mean-d-pb3_lumi_mean': 1, 'f010_pb0_flux_err_mean-d-pb4_flux_err_mean': 1, 'f003_pb2_lumi_median': 0, 'f015_pb0_d0_flux_norm3_q90-m-q10-d-pb5_d0_flux_norm3_q90-m-q10': 1, 'f003_pb2_lumi_q75-d-pb5_lumi_q75': 1, 'f003_pb2_flux_norm1_min-d-pb3_flux_norm1_min': 0, 'f003_pb2_flux_norm2_q75-d-pb4_flux_norm2_q75': 0, 'f003_pb0_flux_norm1_max-d-pb3_flux_norm1_max': 0, 'f015_pb1_d0_lumi_max-m-min': 1, 'f003_pb1_lumi_q75-m-q25-d-pb3_lumi_q75-m-q25': 1, 'f003_pb1_lumi_q90-m-q10-d-pb2_lumi_q90-m-q10': 1, 'f027_lumi_q25': 0, 'f003_pb1_lumi_max': 1, 'fbk1-007_pb0_flux_std': 1, 'f023_pb5_flux_norm3_q75': 0, 'f027_lumi_kurt': 0, 'f003_pb2_flux_norm1_skew': 0, 'f017_pb1_lumi_q90-d-pb3_lumi_q90': 1, 'f003_pb0_lumi_max-m-min-d-pb5_lumi_max-m-min': 0, 'f025_pb2_flux_by_flux_ratio_sq_min': 1, 'f010_pb4_lumi_median-d-pb5_lumi_median': 0, 'fbk1-006_pb1_flux_q75': 0, 'f005_pb1_y2_lumi_max-d-min': 0, 'f023_pb3_lumi_mean-d-pb5_lumi_mean': 1, 'f003_pb2_lumi_q75-d-pb4_lumi_q75': 1, 'f003_pb0_lumi_max-m-min': 0, 'f015_pb1_d1_flux_norm1_max-d-pb3_d1_flux_norm1_max': 1, 'f017_pb2_lumi_q75-d-pb4_lumi_q75': 0, 'f023_pb5_flux_norm1_median': 1, 'f003_pb1_lumi_skew-d-pb3_lumi_skew': 1, 'f024_pb5_flux_q10': 0, 'f003_pb0_lumi_std-d-pb5_lumi_std': 1, 'f014_d1_lumi_q90-m-q10': 0, 'f002_flux_by_flux_ratio_sq_median': 1, 'f025_pb2_lumi_q10-d-pb3_lumi_q10': 1, 'f003_pb0_lumi_median-d-pb1_lumi_median': 1, 'f003_pb1_lumi_q75-d-pb4_lumi_q75': 1, 'f015_pb2_d0_flux_err_q25-d-pb5_d0_flux_err_q25': 1, 'f010_pb0_flux_ratio_sq_median-d-pb4_flux_ratio_sq_median': 1, 'f023_pb3_lumi_q75-d-pb5_lumi_q75': 0, 'f017_pb4_lumi_max-d-min': 1, 'f005_pb5_y2_lumi_median': 0, 'f013_lumi_q90-m-q10': 0, 'f003_pb1_flux_err_min-d-pb3_flux_err_min': 0, 'f023_pb2_flux_by_flux_ratio_sq_median': 0, 'f003_pb4_flux_kurt': 1, 'f014_d1_flux_norm1_std': 1, 'f023_pb4_lumi_q25-d-pb5_lumi_q25': 0, 'f012_pb3_lumi_min-d-pb4_lumi_min': 1, 'f003_pb2_flux_ratio_sq_median-d-pb5_flux_ratio_sq_median': 1, 'f015_pb3_d1_flux_max-d-pb5_d1_flux_max': 0, 'f017_pb5_flux_norm1_q90': 1, 'f017_pb2_lumi_q25-d-pb4_lumi_q25': 0, 'f017_pb1_flux_err_max': 1, 'f003_pb1_lumi_max-d-pb5_lumi_max': 0, 'f023_pb2_lumi_std': 1, 'f003_pb0_flux_norm2_std-d-mean-d-pb4_flux_norm2_std-d-mean': 1, 'f015_pb2_d1_lumi_max-d-pb5_d1_lumi_max': 1, 'f012_pb2_flux_by_flux_ratio_sq_min': 1, 'f003_pb2_lumi_q75-m-q25-d-pb3_lumi_q75-m-q25': 0, 'f005_pb0_y2_flux_norm2_std-d-mean': 0, 'f003_pb0_flux_ratio_sq_q90-d-pb5_flux_ratio_sq_q90': 0, 'f013_lumi_std': 0, 'f003_pb0_lumi_skew': 1, 'f018_pb1_lumi_median-d-pb2_lumi_median': 0, 'f020_pb4_lumi_min': 1, 'f026_pb3_date_diff_0.3': 0, 'f005_pb5_y2_flux_norm2_std-d-mean': 0, 'f003_pb2_lumi_mean-d-pb4_lumi_mean': 1, 'f014_d0_flux_ratio_sq_q90': 0, 'f015_pb0_d0_lumi_q90-d-pb2_d0_lumi_q90': 1, 'f013_lumi_q75-m-q25': 0, 'f003_pb1_lumi_max-d-pb3_lumi_max': 0, 'f004_y0_lumi_q25-d-y1_lumi_q25': 0, 'f002_flux_norm1_q90': 0, 'f003_pb0_flux_norm1_median': 0, 'fbk1-006_pb0_flux_mean': 1, 'f015_pb5_d0_flux_norm3_std-d-mean': 0, 'f012_pb5_flux_norm2_mean': 0, 'f003_pb0_flux_ratio_sq_q75-d-pb4_flux_ratio_sq_q75': 1, 'f018_pb1_lumi_median-d-pb4_lumi_median': 1, 'f017_pb2_flux_norm2_mean-d-pb4_flux_norm2_mean': 1, 'f015_pb1_d1_flux_by_flux_ratio_sq_max-d-min': 0, 'f023_pb5_flux_mean': 1, 'f003_pb1_lumi_q75-d-pb3_lumi_q75': 0, 'f003_pb0_lumi_std-d-mean-d-pb1_lumi_std-d-mean': 1, 'f015_pb0_d0_lumi_kurt-d-pb1_d0_lumi_kurt': 1, 'f003_pb0_lumi_mean-d-pb1_lumi_mean': 1, 'f023_pb4_flux_norm3_q90': 1, 'f017_pb3_flux_norm1_min': 1, 'f003_pb2_lumi_max-d-pb4_lumi_max': 0, 'f010_pb4_lumi_std-d-mean-d-pb5_lumi_std-d-mean': 0, 'f010_pb2_flux_err_median-d-pb3_flux_err_median': 1})
"""

#==============================================================================
utils.end(__file__)
utils.stop_instance()


