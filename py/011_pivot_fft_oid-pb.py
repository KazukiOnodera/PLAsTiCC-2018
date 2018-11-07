#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:52:16 2018

@author: Kazuki


keys: object_id, passband

"""


import numpy as np
import pandas as pd
import os
from glob import glob
from multiprocessing import cpu_count, Pool
from itertools import combinations
import utils

PREF = 'f011'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')

def quantile(n):
    def quantile_(x):
        return np.percentile(x, n)
    quantile_.__name__ = 'q%s' % n
    return quantile_

threshold = 0.5
def fft(n):
    def fft_(x):
        N = x.shape[0]
        # 高速フーリエ変換(FFT)
        F = np.fft.fft(x)
        # FFT結果（複素数）を絶対値に変換
        F_abs = np.abs(F)
        # 振幅を元に信号に揃える
        F_abs_amp = F_abs / N * 2 # 交流成分はデータ数で割って2倍する
        F_abs_amp[0] = F_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要
        tmp = F_abs_amp[:int(N/2)+1]
        amp = tmp[tmp>threshold]
        try:
            peri = np.where(tmp == amp[n])[0][0]
        except:
            return None, None
        return peri, amp[n]
    fft_.__name__ = 'fft%s' % n
    return fft_

def fft_max(x):
    N = x.shape[0]
    # 高速フーリエ変換(FFT)
    F = np.fft.fft(x)
    # FFT結果（複素数）を絶対値に変換
    F_abs = np.abs(F)
    # 振幅を元に信号に揃える
    F_abs_amp = F_abs / N * 2 # 交流成分はデータ数で割って2倍する
    F_abs_amp[0] = F_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要
    F_abs_amp = F_abs_amp[:int(N/2)+1]
    try:
        peri = F_abs_amp.argmax()
    except:
        return None, None
    return peri, max(F_abs_amp)


num_aggregations = {
    'flux':        [fft_max, fft(0), fft(1)],
    'flux_norm1':  [fft_max, fft(0), fft(1)],
    'flux_norm2':  [fft_max, fft(0), fft(1)],
    'flux_err':    [fft_max, fft(0), fft(1)],
    }

def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    pt = pd.pivot_table(df, index=['object_id'], columns=['passband'], 
                        aggfunc=num_aggregations)
    
    pt.columns = pd.Index([f'pb{e[2]}_{e[0]}_{e[1]}' for e in pt.columns.tolist()])
    
    col = pt.columns
    for c in col:
        pt[f'{c}_peri'] = pt[c].map(lambda x: x[0])
        pt[f'{c}_amp'] = pt[c].map(lambda x: x[1])
    pt.drop(col, axis=1, inplace=True)
    
    # compare passband
    col = pd.Series([f'{c[3:]}' for c in pt.columns if c.startswith('pb0')])
    for c1,c2 in list(combinations(range(6), 2)):
        col1 = (f'pb{c1}'+col).tolist()
        col2 = (f'pb{c2}'+col).tolist()
        for c1,c2 in zip(col1, col2):
            pt[f'{c1}-d-{c2}'] = pt[c1] / pt[c2]
    
    if drop_oid:
        pt.reset_index(drop=True, inplace=True)
    else:
        pt.reset_index(inplace=True)
    pt.add_prefix(PREF+'_').to_pickle(output_path)
    
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
    
    aggregate(pd.read_pickle('../data/train_log.pkl'), f'../data/train_{PREF}.pkl')
    
    # test
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
    df.to_pickle(f'../data/test_{PREF}.pkl')
    os.system(f'rm ../data/tmp_{PREF}*')
    
    utils.end(__file__)


