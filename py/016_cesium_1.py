#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:58:35 2018

@author: Kazuki

keys: object_id, passband

"""

import numpy as np
import pandas as pd
import os
from glob import glob
from multiprocessing import cpu_count, Pool
import cesium.featurize as featurize
import utils

PREF = 'f016'

os.system(f'rm ../data/t*_{PREF}*')
os.system(f'rm ../feature/t*_{PREF}*')


features_to_use=['freq1_freq',
                'freq1_signif',
                'freq1_amplitude1',
                'skew',
                'percent_beyond_1_std',
                'percent_difference_flux_percentile']


def normalise(ts):
    return (ts - ts.mean()) / ts.std()


def aggregate(df, output_path, drop_oid=True):
    """
    df = pd.read_pickle('../data/train_log.pkl').head(999)
    """
    
    df.mjd  = df.mjd.astype(float)
    df.flux = df.flux.astype(float)
    
    groups = df.groupby(['object_id', 'passband'])
    times = groups.apply(
        lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
    flux = groups.apply(
        lambda block: normalise(block['flux']).values
    ).reset_index().rename(columns={0: 'seq'})
#    err = groups.apply(
#        lambda block: (block['flux_err'] / block['flux'].std()).values
#    ).reset_index().rename(columns={0: 'seq'})
#    det = groups.apply(
#        lambda block: block['detected'].astype(bool).values
#    ).reset_index().rename(columns={0: 'seq'})
    
    oid = df.groupby('object_id').object_id.mean().values
    
    times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
    flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
#    err_list = err.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
#    det_list = det.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
    
    feats = featurize.featurize_time_series(times=times_list,
                                            values=flux_list,
                                            features_to_use=features_to_use,
                                            scheduler=None)
    
    tmp = pd.DataFrame()
    for c in features_to_use:
        tmp[f'{c}_min'] = feats[c].min(1) 
        tmp[f'{c}_mean'] = feats[c].mean(1) 
        tmp[f'{c}_max'] = feats[c].max(1) 
        tmp[f'{c}_std'] = feats[c].std(1) 
    
    feats.columns = pd.Index([f'pb{e[1]}_{e[0]}' for e in feats.columns.tolist()])
    
    feats = pd.concat([feats, tmp], axis=1)
    feats['object_id'] = oid
    
    
#    # std / mean
#    col_std = [c for c in feats.columns if c.endswith('_std')]
#    for c in col_std:
#        feats[f'{c}-d-mean'] = feats[c]/feats[c.replace('_std', '_mean')]
    
    # max / min, max - min
    col_max = [c for c in feats.columns if c.endswith('_max')]
    for c in col_max:
        feats[f'{c}-d-min'] = feats[c]/feats[c.replace('_max', '_min')]
        feats[f'{c}-m-min'] = feats[c]-feats[c.replace('_max', '_min')]
    
    if drop_oid:
        feats.reset_index(drop=True, inplace=True)
    else:
        feats.reset_index(inplace=True)
    feats.add_prefix(PREF+'_').to_pickle(output_path)
    
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
    utils.stop_instance()


