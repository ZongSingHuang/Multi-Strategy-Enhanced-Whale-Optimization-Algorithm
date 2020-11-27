# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:26:44 2020

@author: ZongSing_NB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def import_data(data_name, period, rolling):
    data_set = []
    for i in range(len(data_name)):
        # 匯入
        data = pd.read_table(data_name[i], header=None)
        # 檢查
        try:
            data = data[[3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        except:
            print('資料不足43行，無法匯入')
        # 平滑
        data = data.rolling(int(rolling)).mean()
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        # 取樣
        data = data[data.index%(int(period))==0]
        data.reset_index(drop=True, inplace=True)
        
        # 歸零
        data[[3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = data[[3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] - data[[3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].iloc[0]
        # 保存
        data_set.append(data.values)
    
    # 合併
    new_data_set = np.array(data_set[0])
    for i in range(len(data_set)-1):
        data = np.array(data_set[i+1])
        new_data_set = np.vstack([new_data_set, data])
    
    # 拆分
    x = new_data_set[:, 1:]
    y = new_data_set[:, 0]
    
    return x, y

def sc(x_tr, y_tr, x_ts, y_ts):
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_tr_sc = sc_x.fit_transform(x_tr)
    y_tr_sc = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    x_ts_sc = sc_x.transform(x_ts)
    y_ts_sc = sc_y.transform(y_ts.reshape(-1, 1)).ravel()
    
    return x_tr_sc, y_tr_sc, x_ts_sc, y_ts_sc, sc_x, sc_y

def plot_fig(actual, pred_y, residual, title):
    plt.figure()
    plt.title(title)
    plt.plot(actual, label='actual', color='black')
    plt.plot(pred_y, label='predict', color='red')
    plt.plot(residual, label='residual', color='blue')
    plt.xlabel('Sampling time')
    plt.ylabel('Displacement(μm)')
    plt.legend()
    plt.xticks()
    plt.yticks()
    plt.grid()
    plt.show()

def test(model, x_ts_sc, y_ts_sc, y_ts, best_inputs, sc_y):
    # 預測
    if best_inputs is not None:
        pred_y = model.predict(x_ts_sc[:, best_inputs])
    else:
        pred_y = model.predict(x_ts_sc)
    if pred_y.ndim!=1:
        pred_y = pred_y.reshape(-1)
    # 反正規化
    if sc_y is not None:
        pred_y = sc_y.inverse_transform(pred_y)
    # 誤差
    residual_ts = y_ts - pred_y
    rmse_ts = mean_squared_error(y_ts, pred_y, squared=False)
    step_ts = np.max(residual_ts) - np.min(residual_ts)
    
    return pred_y, residual_ts, rmse_ts, step_ts

def difference(data_x, data_y, step=5):
    new_x = data_x[step:, :].copy()
    for i in range(step):
        new_x = np.hstack([new_x, np.diff(data_x, n=i+1, axis=0)[step-(i+1):, :]])
        
    new_y = data_y[step:]
    return new_x, new_y
# d0 = data_x[:, feature].copy()
# d1 = np.diff(d0, n=1, axis=0)
# d2 = np.diff(d0, n=2, axis=0)
# d3 = np.diff(d0, n=3, axis=0)
# d4 = np.diff(d0, n=4, axis=0)
# d5 = np.diff(d0, n=5, axis=0)
# new = np.hstack([d0[5:, :], d1[4:, :], d2[3:, :], d3[2:, :], d4[1:, :], d5])
# new, data_y[5:]