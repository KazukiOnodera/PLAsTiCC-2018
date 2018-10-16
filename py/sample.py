#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:55:54 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import utils, os


os.system('rm -rf ../sample')
os.system('mkdir ../sample')


tr  = utils.load_train()
log = pd.read_feather('../data/train_log.f')

oids = tr.sample(999).object_id.tolist()

tr_ = tr[tr.object_id.isin(oids)]
log_ = log[log.object_id.isin(oids)]

tr_.to_csv('../sample/tr.csv', index=False)
log_.to_csv('../sample/tr_log.csv', index=False)

