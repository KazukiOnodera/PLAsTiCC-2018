#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:11:50 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils, utils_post, utils_metric
#utils.start(__file__)

import optuna

from multiprocessing import cpu_count, Pool

# =============================================================================
# weight
# =============================================================================
oof = pd.read_pickle('../FROM_MYTEAM/oof_v103_068_lgb__v103_062_nn__specz_avg.pkl')
oof = oof.copy().values.astype(float)

y = utils.load_target().target
y_ohe = pd.get_dummies(y)

weight = utils_post.get_weight(y_ohe, oof, eta=0.1, nround=9999)

print(f'weight: np.array({list(weight)})')


weight = utils_post.get_weight(y_ohe, oof, eta=0.1, nround=9999, based_true=False)

print(f'weight: np.array({list(weight)})')

# =============================================================================
# one by one
# =============================================================================


oof = pd.read_pickle('../FROM_MYTEAM/oof_v103_068_lgb__v103_062_nn__specz_avg.pkl').reset_index(drop=True)

oid_gal = pd.read_pickle('../data/tr_oid_gal.pkl')['object_id'].tolist()
oid_exgal = pd.read_pickle('../data/tr_oid_exgal.pkl')['object_id'].tolist()

classes_gal = [6, 16, 53, 65, 92]
classes_exgal = [15, 42, 52, 62, 64, 67, 88, 90, 95]


sub_tr = utils.load_train(['object_id'])

sub_tr = pd.concat([sub_tr, oof], axis=1)
sub_tr.columns = ['object_id'] +[f'class_{i}' for i in sorted(classes_gal+classes_exgal)]


sub_tr.loc[sub_tr.object_id.isin(oid_gal),  [f'class_{i}' for i in classes_exgal]] = 0
sub_tr.loc[sub_tr.object_id.isin(oid_exgal),[f'class_{i}' for i in classes_gal]] = 0

#weight = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
#weight = weight / sub_tr.iloc[:,1:].sum()
#weight = weight.values
#
oof = sub_tr.iloc[:,1:]#.values.astype(float)

y = utils.load_target().target
y_ohe = pd.get_dummies(y)

weight = utils_post.get_weight(y_ohe, oof, eta=0.1, nround=9999, based_true=False)



def multi_weighted_logloss(y_ohe:np.array, y_preds:np.array):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
#    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
#    if len(np.unique(y_true)) > 14:
#        classes.append(99)
#        class_weight[99] = 2
        
    y_p = y_preds/y_preds.sum(1)[:,None]
    # Trasform y_true in dummies
    y_ohe = pd.DataFrame(y_ohe)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

weight = utils_post.get_weight(y_ohe, oof.values, eta=0.1, nround=9999)

weight = np.array([0.93518041791149, 1.0371997389092826, 0.7276439651240076, 
                   0.6327491122383724, 1.1299087345271177, 1.5915423992967905, 
                   0.7171293513633095, 1.4088185214955187, 0.5753104825120527, 
                   1.0969786180736176, 0.8075063922456374, 0.5918521326189743, 
                   1.1820632733042782, 0.9285728506117077])

oof = oof * weight

print(multi_weighted_logloss(y_ohe, oof.values)) # 0.43791792929663814




oof_ = oof.copy()
for c in oof.columns:
#    oof_.loc[oof_[c] < 0.0000001, c] = 0
    oof_.loc[oof_[c] > 0.99, c] = 0.95


print(multi_weighted_logloss(y_ohe, oof_.values))




# =============================================================================
# 
# =============================================================================


def objective(trial):
    weight = []
    weight.append( trial.suggest_discrete_uniform('c6', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c15', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c16', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c42', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c52', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c53', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c62', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c64', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c65', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c67', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c88', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c90', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c92', 0.00001, 3.0, 0.00001) )
    weight.append( trial.suggest_discrete_uniform('c95', 0.00001, 3.0, 0.00001) )
    weight = np.array(weight)
    
    return multi_weighted_logloss(y_ohe, oof * weight)

y = utils.load_target().target
y_ohe = pd.get_dummies(y)

study = optuna.create_study()
study.optimize(objective, n_trials=9999)








#==============================================================================
utils.end(__file__)
