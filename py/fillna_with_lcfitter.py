#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:26:14 2018

@author: kazuki.onodera
"""


import numpy as np
import pandas as pd
import os
from glob import glob
#from scipy.stats import kurtosis
from multiprocessing import cpu_count, Pool

import sys
argvs = sys.argv

from itertools import combinations
import utils



opt_param_df = pd.read_csv('../FROM_MYTEAM/sn_template_params_20181129.csv')

tr = utils.load_train(['object_id', 'hostgal_photoz'])
te = utils.load_test(['object_id', 'hostgal_photoz'])

exgal_oids = tr[tr.hostgal_photoz>0].object_id.tolist() + te[te.hostgal_photoz>0].object_id.tolist()






# 超新星LCにフィットする関数(0-2でclippingしておく)
def LC_fit_func(x, t0, a, b, c, Tf, Tr):
    return np.clip(a*(1+b*(x-t0)+c*(x-t0)*(x-t0))*np.exp(-(x-t0)/Tf) / (1+np.exp(-(x-t0)/Tr)), a_min=0, a_max=2)

# fittingのχ二乗を求める関数
def get_chisq(x, obs_flux, obs_flux_err, opt_param):
    template_flux = LC_fit_func(x, *opt_param)
    
    # calculate chi-square
    residual = obs_flux - template_flux
    dof = len(residual) - 2 # 横移動と、縦の伸縮の2パラメータでfitしているので観測個数から2を引く
    chisq = np.square(residual / obs_flux_err).sum()/dof
    return chisq

def LC_single_fit(df_, opt_param, d_range=[-30, 30], d_freq=5, flux_range=[1.0, 4.0], flux_freq=0.1):
    '''
    観測データ（df_）にLCを当てはめる
    
    input
        - df_:1天体の観測データ（singleバンドのfitを想定してるので事前にpassbandを絞っておく）
        - opt_param: 当てはめたいLCテンプレートのパラメータ（ndarray）
        - d_range: 観測されてる最大光度日から前後何日を探索するか
        - d_freq: 最大光度日の探索間隔
        - flux_range: 最大光度の探索範囲（log10なので、例えば2はfluxで10**2=100を意味する）
        - flux_freq: 最大光度の探索間隔
        
    output:
        - chisq_list: 各探索毎のχ二乗を入れたndarray
        - min_chisq: 一番小さいχ二乗値
        - f_opt:  一番当てはまりの良かった最大光度（log10(flux)単位）
        - d_opt: 一番当てはまりの良かった（観測）最大光度日からのday差分
        - search_params: 探索範囲の次元(d 方向の探索回数、f方向の探索回数, d_range[0], d_freq, flux_range[0], flux_freq)
    '''   
    # define serach range (max_mjd)
    search_max_mjd_series_ = np.arange(d_range[0], d_range[1]+d_freq, d_freq)
    # define serach range (max_flux)
    search_flux_series_ = np.arange(flux_range[0], flux_range[1]+flux_freq, flux_freq)
    
    # template fit（d, fの探索範囲ごとにχ二乗を計算して格納）
    chisq_list = []
    for d_ in search_max_mjd_series_:
        x = df_.mjd_from_max.values - d_
        for f_ in search_flux_series_:
            f_corr_factor = 10.0**f_
            y = df_.flux.values / f_corr_factor
            yerr = df_.flux_err.values / f_corr_factor
            chisq_ = get_chisq(x, y, yerr, opt_param)
            chisq_list.append(chisq_)
            
    chisq_list = np.array(chisq_list)
    min_chisq = np.min(chisq_list) # 最小のχ二乗
    
    min_index = np.argmin(chisq_list)  #最小のχ二乗のindex
    d_opt = d_range[0] + d_freq * (min_index // len(search_flux_series_)) # χ二乗を最小にするd
    f_opt = flux_range[0] + flux_freq *(min_index % len(search_flux_series_)) # χ二乗を最小にするf（※log10単位）
    search_params = (len(search_max_mjd_series_), len(search_flux_series_), d_range[0], d_freq, flux_range[0], flux_freq)
    
    return chisq_list, min_chisq, d_opt, f_opt, search_params

X = np.arange(-400, 400, 5)

def __get_feature__(df, param):
    """
    input:
        df = 713
        param = (target=90; passband=4; zrange=0)
    
    output:
        y: array
        min(chisq_list): array
        d_opt: float
        f_opt: float
        search_params: tuple(6 dimensions)
        
    """
    
    chisq_list, min_chisq, d_opt, f_opt, search_params = LC_single_fit(df, param)
    y = LC_fit_func(X-d_opt, *param)* 10.0**f_opt
    
    return y, min(chisq_list), d_opt, f_opt, search_params

targets = [42, 52, 62, 67, 90]
zranges = [0,1,2,3]

def get_feature(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl')
    """
    
    ts_df = pd.DataFrame()
    passband = 4
    df = df.loc[(df.passband==passband) & (df.object_id.isin(exgal_oids))]
    
    
    oids = df.object_id.unique()
    for oid in oids:
        df_ = df[df.object_id==oid]
        ts_df_ = pd.DataFrame(X, columns='date')
        ts_df_['object_id'] = oid
        
        for target in targets:
            
            for zrange in zranges:
                param = opt_param_df[(opt_param_df.target==target) & (opt_param_df.passband==passband) & (opt_param_df.zrange==zrange)][['t0', 'a', 'b', 'c', 'Tf', 'Tr']].values[0]
                
                y, chisq_min, d_opt, f_opt, search_params = __get_feature__(df_, param)
                ts_df_[f'c{target}_z{zrange}'] = y
    
    
    
    if drop_oid:
        feature.reset_index(drop=True, inplace=True)
    else:
        feature.reset_index(inplace=True)
    feature.add_prefix(PREF+'_').to_pickle(output_path)
    
    return

def multi(args):
    input_path, output_path = args
    aggregate(pd.read_pickle(input_path), output_path, drop_oid=False)
    return

# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    utils.start(__file__)
    
    usecols = None
    aggregate(pd.read_pickle('../data/train_log.pkl'), f'../data/train_{PREF}.pkl')
    
    if utils.GENERATE_AUG:
        os.system(f'rm ../data/tmp_{PREF}*')
        argss = []
        for i,file in enumerate(utils.AUG_LOGS):
            argss.append([file, f'../data/tmp_{PREF}{i}.pkl'])
        pool = Pool( cpu_count() )
        pool.map(multi, argss)
        pool.close()
        df = pd.concat([pd.read_pickle(f) for f in glob(f'../data/tmp_{PREF}*')], 
                        ignore_index=True)
        df.sort_values(f'{PREF}_object_id', inplace=True)
        df.reset_index(drop=True, inplace=True)
        del df[f'{PREF}_object_id']
        utils.to_pkl_gzip(df, f'../data/train_aug_{PREF}.pkl')
        os.system(f'rm ../data/tmp_{PREF}*')
    
    # test
    if utils.GENERATE_TEST:
        imp = pd.read_csv(utils.IMP_FILE).head(utils.GENERATE_FEATURE_SIZE)
        usecols = imp[imp.feature.str.startswith(f'{PREF}')][imp.gain>0].feature.tolist()
#        usecols = [c.replace(f'{PREF}_', '') for c in usecols]
#        usecols += ['object_id']
        
        os.system(f'rm ../data/tmp_{PREF}*')
        argss = []
        for i,file in enumerate(utils.TEST_LOGS):
            argss.append([file, f'../data/tmp_{PREF}{i}.pkl'])
        pool = Pool( cpu_count() )
        pool.map(multi, argss)
        pool.close()
        df = pd.concat([pd.read_pickle(f) for f in glob(f'../data/tmp_{PREF}*')], 
                        ignore_index=True)
        df.sort_values(f'{PREF}_object_id', inplace=True)
        df.reset_index(drop=True, inplace=True)
        del df[f'{PREF}_object_id']
        utils.to_pkl_gzip(df, f'../data/test_{PREF}.pkl')
        utils.save_test_features(df[usecols])
        os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)



