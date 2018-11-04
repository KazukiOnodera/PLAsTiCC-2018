#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:30:33 2018

@author: Kazuki
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

N = 64 # データ数
n = np.arange(N)
f1 = 4 # 周期①
f2 = 10 # 周期②
a1 = 1.5 # 振幅①
a2 = 3 # 振幅②
f = a1 * np.sin(f1 * 2 * np.pi * (n/N)) + a2 * np.sin(f2 * 2 * np.pi * (n/N)) 

# グラフ表示
plt.figure(figsize=(8, 4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.plot(f)


# =============================================================================
# 
# =============================================================================
# 高速フーリエ変換(FFT)
F = np.fft.fft(f)
# FFT結果（複素数）を絶対値に変換
F_abs = np.abs(F)
# 振幅を元に信号に揃える
F_abs_amp = F_abs / N * 2 # 交流成分はデータ数で割って2倍する
F_abs_amp[0] = F_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要

# グラフ表示（データ数の半分の周期を表示）
plt.plot(F_abs_amp[:int(N/2)+1])

