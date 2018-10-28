#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 02:44:42 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)

FILE_in  = '../output/1022-1.csv.gz'

sub = pd.read_csv(FILE_in)

oid_gal   = pd.read_pickle('../data/oid_gal.pkl').object_id
oid_exgal = pd.read_pickle('../data/oid_exgal.pkl').object_id


# =============================================================================
# Giba post
# =============================================================================
"""
Giba says

My LB probing experiments, up to now, showed me that 
distribution of class_99 Galactic is around 10x smaller than class_99 ExtraGalactic. 
Hard coding class_99 of Galactic and ExtraGalactic to constants 0.017 and 0.17 
improved my scores.

"""

FILE_out = '../output/1022-1_Giba-post.csv.gz'

sub.loc[sub.object_id.isin(oid_gal),   'class_99'] = 0.017
sub.loc[sub.object_id.isin(oid_exgal), 'class_99'] = 0.17

sub.to_csv(FILE_out, index=False, compression='gzip')
utils.submit(FILE_out, '0.017 and 0.17')

# =============================================================================
# yuval post
# =============================================================================
"""
yuval says

I believe a prerequisite for predicting Class 99 is having a very good model 
for the other classes. The good new is that a score of ~1.0 is achievable 
without really dealing with class 99. The only thing I did up to this point 
with this prediction is: class_99=np.where(other_classes.max>0.9 , 0.01, 0.1) 
[it is slightly better then uniform, If I use 0.8 the score degrades, 
and also other values of probabilities degrade the score]


"""

sub = pd.read_csv(FILE_in)

FILE_out = '../output/1022-1_yuval-post.csv.gz'

sub['class_99'] = 0
class_99 = np.where(sub.iloc[:,1:].max(1)>0.9 , 0.01, 0.1)

sub.to_csv(FILE_out, index=False, compression='gzip')
utils.submit(FILE_out, 'max>0.9 , 0.01, 0.1')


#==============================================================================
utils.end(__file__)
