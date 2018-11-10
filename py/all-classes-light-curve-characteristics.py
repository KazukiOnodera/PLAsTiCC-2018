#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:31:33 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import chain
sns.set_style('whitegrid')
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
import cesium.featurize as featurize


train_series = pd.read_csv('../input/training_set.csv.zip')
train_metadata = pd.read_csv('../input/training_set_metadata.csv')

def normalise(ts):
    return (ts - ts.mean()) / ts.std()

groups = train_series.groupby(['object_id', 'passband'])
times = groups.apply(
    lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
flux = groups.apply(
    lambda block: normalise(block['flux']).values
).reset_index().rename(columns={0: 'seq'})
err = groups.apply(
    lambda block: (block['flux_err'] / block['flux'].std()).values
).reset_index().rename(columns={0: 'seq'})
det = groups.apply(
    lambda block: block['detected'].astype(bool).values
).reset_index().rename(columns={0: 'seq'})

times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
err_list = err.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
det_list = det.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()


stds = groups['flux'].std().groupby('object_id').median()
unique_classes = train_metadata['target'].unique()

# =============================================================================
# def
# =============================================================================
def plot_phase(n, fr=None, extra_title='', hide_undetected=False, ax=None):
    selected_times = times_list[n]
    selected_flux = flux_list[n]
    selected_err = err_list[n]
    if hide_undetected:
        selected_det = det_list[n]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('time' if fr is None else 'phase')
    for band in range(6):
        if hide_undetected:
            times = selected_times[band][selected_det[band]] \
                if fr is None else (selected_times[band][selected_det[band]] * fr) % 1
            ax.scatter(x=times, 
                       y=selected_flux[band][selected_det[band]], 
                       c=colors[band],
                       s=10)
            ax.vlines(times, 
                      selected_flux[band][selected_det[band]] 
                      - selected_err[band][selected_det[band]],
                      selected_flux[band][selected_det[band]] 
                      + selected_err[band][selected_det[band]],
                      colors=colors[band],
                      linewidth=1)
            
            times = selected_times[band][~selected_det[band]] \
                if fr is None else (selected_times[band][~selected_det[band]] * fr) % 1
            ax.scatter(x=times, 
                       y=selected_flux[band][~selected_det[band]], 
                       c=colors[band],
                       alpha=0.3,
                       s=5)
            ax.vlines(times, 
                      selected_flux[band][~selected_det[band]] 
                      - selected_err[band][~selected_det[band]],
                      selected_flux[band][~selected_det[band]] 
                      + selected_err[band][~selected_det[band]],
                      colors=colors[band],
                      alpha=0.1,
                      linewidth=0.5)
        else:
            times = selected_times[band] if fr is None else (selected_times[band] * fr) % 1
            ax.scatter(x=times, 
                       y=selected_flux[band], 
                       c=colors[band],
                       s=10)
            ax.vlines(times, 
                      selected_flux[band] - selected_err[band],
                      selected_flux[band] + selected_err[band],
                      colors=colors[band],
                      linewidth=1)
    ax.set_ylabel('relative flux')
    ax.set_title(
        f'object: {train_metadata["object_id"][n]}, class: {train_metadata["target"][n]}, '
        f'ddf: {train_metadata["ddf"][n]}, specz: {train_metadata["hostgal_specz"][n]}'
        f', median flux std: {stds[train_metadata["object_id"][n]]:.4}'
        + extra_title)
    if ax is None:
        plt.show()

def get_freq_features(N, subsetting_pos=None):
    if subsetting_pos is None:
        subset_times_list = times_list
        subset_flux_list = flux_list
    else:
        subset_times_list = [v for i, v in enumerate(times_list) 
                             if i in set(subsetting_pos)]
        subset_flux_list = [v for i, v in enumerate(flux_list) 
                            if i in set(subsetting_pos)]
    feats = featurize.featurize_time_series(times=subset_times_list[:N],
                                            values=subset_flux_list[:N],
                                            features_to_use=['freq1_freq',
                                                            'freq1_signif',
                                                            'freq1_amplitude1',
                                                            'skew',
                                                            'percent_beyond_1_std',
                                                            'percent_difference_flux_percentile'],
                                            scheduler=None)
    feats['object_pos'] = subsetting_pos[:N]
    return feats

def get_class_feats(label, N=20):
    class_pos = train_metadata[train_metadata['target'] == label].index
    class_feats = get_freq_features(N, class_pos)
    return class_feats


def plot_phase_curves(feats, use_median_freq=False, hide_undetected=True, N=10):
    for i in range(N):
        freq = feats.loc[i, 'freq1_freq'].median()
        freq_min = feats.loc[i, 'freq1_freq'].min()
        freq_std = feats.loc[i, 'freq1_freq'].std()
        skew = feats.loc[i, 'skew'].mean()
        object_pos = int(feats.loc[i, 'object_pos'][0])
        f, ax = plt.subplots(1, 2, figsize=(14, 4))
        plot_phase(object_pos, 
                   fr=None,
                   hide_undetected=hide_undetected, ax=ax[0])
        plot_phase(object_pos, 
                   fr=freq if use_median_freq else freq_min,
                   hide_undetected=hide_undetected, ax=ax[1])
        title = ax[0].get_title()
        ax[0].set_title('time')
        ax[1].set_title('phase')
        f.suptitle(title + f'\nfreq_std: {freq_std:.2}'
                   f', median period: {1/freq: .2}, max period: {1/freq_min: .2}'
                   f', mean skew: {skew:.2}', y=1.1)
        plt.show()


# =============================================================================
# 
# =============================================================================
class92_feats = get_class_feats(92)
plot_phase_curves(class92_feats)

