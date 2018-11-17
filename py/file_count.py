#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 10:15:23 2018

@author: Kazuki
"""

from glob import glob
from collections import Counter

files = [f.split('_')[1] for f in glob('../feature/*')]


Counter(files)


