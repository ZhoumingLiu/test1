# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 01:33:39 2022

@author: chais
"""

import os
os.chdir('C:/Users/chais/OneDrive/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts')
# os.chdir('D:/OneDrive/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts')
# os.chdir('C:/Users/SJ CHAI/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts')
# os.chdir('C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts')
# os.chdir('D:/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts')

# import MySqlUtil
# import MongoUtil
# import RedisUtil

# from datetime import datetime,timedelta
# import sys
# import json
import pandas as pd
import importlib
import numpy as np
# import time
# import torch
# from torch import nn
# import threading

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
# from utils_lift import *
# from utils_lift_VTCL4 import *

from utils_OnlineTest import *

from paras_lift import paras_TEST
from utils_lift import *

# from utils_lift_QEBL8 import *


# os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/Utils/")
# from utils_ErgatianLifts import * 


#%% 选择电梯并读取前1min数据


dateStr = '2024-11-28 11:00:00'


paras = paras_TEST

Lift_ID = paras['Lift_ID']

df, data = Load_Build_DataFrame(dateStr, paras, Lift_ID)

# df['Time'] = pd.to_datetime(df['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
# df = df.set_index('Time') 
# dataset = df.resample('S').first()
# dataset.fillna(method='ffill', inplace=True)

print(df)    
# df.to_csv('out.csv')
#%% 触发实时计算脚本

module = importlib.import_module('devpy' + data[4])
operation_class = getattr(module, "runMethod")
result,CarSeg_Stats,DoorSeg_Stats = operation_class(df)
print(result)    

#%% 选择电梯并计算 - hourly脚本

# dateStr = '2023-10-20 11:00:39'

# paras = paras_EMSDHQL11

# Lift_ID = paras['Lift_ID']

# df, data = Load_Build_DataFrameHourly(dateStr, paras, Lift_ID)
# module = importlib.import_module('devpy' + data[4])
# operation_class = getattr(module, "runMethod_hourly")
# result,CarSeg_Stats,DoorSeg_Stats = operation_class(df)
# print(result)    

#%% 数据切分和预处理

df['Time'] = pd.to_datetime(df['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
df = df.set_index('Time') 

df['Velocity'] = np.abs(df['Distance'].diff(periods=20))  # calculate the abs velocity
df = df.interpolate().fillna(method='bfill') # 用插值法补全缺失数据 
# df['Time'] = pd.to_datetime(df['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
# df = df.set_index('Time')
# dataset_raw = dataset_raw[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance']]
df = df[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq',
        'FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag',
        'Distance','floor','velocity','height','mileage','openCloseDoorNum','cumulativeRunNum',
        'vibration_x','vibration_y','vibration_z','door','workMode']]

# df.loc[df['Brake'] < 0.2,'Brake'] = 0
CarSeg_list, DoorSeg_list = Data_segment(df, paras, 0.5, 1)


# CarStat_list, DoorStat_list, event_list3 = do_action(df,paras, CarSeg_list,DoorSeg_list)

seq = DoorSeg_list[0]

#%% 分析！！！
dataset = df.copy()
# 将时间戳列转换为 DateTimeIndex
dataset['Time'] = pd.to_datetime(dataset['Time'],format='%Y-%m-%d %H:%M:%S:%f')
dataset = dataset.set_index('Time')
# 重采样，选择每秒的第一个数据点
dataset = dataset.resample('S').first()

dataset = dataset.ffill() #补齐楼层，用最近值代替空值

dataset = dataset.sort_index(ascending=False)


##1.计算floor=NAN的长度和丢包率

MissingFloor = np.isnan(dataset['velocity']).values.sum()/len(dataset) 
print(MissingFloor)

##2.查看连续NAN的数量 

##3.画velocity和distance的图

# seq['velocity'] = seq['velocity'].interpolate() #补齐velocity，用均值代替

EgPlot_Dist_Vel(dataset, 'EMSDHQ_L11', 12, 5, 1.5, 2)


#4.画Floor和distance的图
dataset['floor'] = dataset['floor'].ffill() #补齐楼层，用最近值代替空值

# dataset.loc[dataset['floor'] == -1,'floor'] = 0
# dataset.loc[dataset['floor'] == -2,'floor'] = -1
# seq.loc[seq['floor'] == 'NAN','floor'] = np.nan
# seq['floor'] = seq['floor'].astype(float) 

Plot_Dist_Floor(paras, dataset, 'EMSDHQ_L11', 12, 5, 1.5, 2)
                                                                   

#5.画brake和floor的图
Plot_Brake_Floor(paras, dataset, 'EMSDHQ_L11', 12, 5, 1.5, 2)

#%% 滤波处理和画图(sequence)

# df = DistVel_FilterA(df,2,0.05) # 滤波速度和距离 - Method A
# df = DistVel_FilterB(df,0.5,3.3) # 滤波速度和距离 - Method B


# EgPlot_Comparison(df, paras['Lift_Name'], 12, 28, 1.5, 2)   # raw distance  Dist A 和 Dist B对比  

# EgPlot_Distance(df, paras['Lift_Name'], 12, 22, 1.5, 2) # raw distance 和 Dist A d对比  

#%% KF vs. raw data

df['Time'] = pd.to_datetime(df['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
df = df.set_index('Time') 

smoother = KalmanSmoother(component='level_trend',
                      component_noise={'level': 0.1, 'trend': 0.1})

values = df[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance']]
smooth_seq = smoother.smooth(values.T).smooth_data.T # all smoothed sequences, if nan for a whole column, it will be 0.   
df[['Motor_KF','Brake_KF','Safety_KF','Door_KF','Resv1_KF','Resv2_KF','Resv3_KF','Dist_KF']] = smooth_seq

# seq = seq.reset_index()
# seq = seq.rename(columns={'index': 'Time'})
EgPlot_RawVSKF(df['Motor'],df['Motor_KF'], 'WCH WW L9', 12, 5, 1.5, 2)
#%% FFT
dataset = df.copy()
# 将时间戳列转换为 DateTimeIndex
dataset['Time'] = pd.to_datetime(dataset['Time'],format='%Y-%m-%d %H:%M:%S:%f')
dataset = dataset.set_index('Time')
# 重采样，选择每秒的第一个数据点
dataset = dataset.resample('S').first()
FFTMag_plot(dataset, 12, 5, 1.5, 2)
FFTFreq_plot(dataset, 12, 5, 1.5, 2)
FFTFreqMag_plot(dataset, 12, 5, 1.5, 2)

#%% Old feature plot
dataset = df.copy()
# 将时间戳列转换为 DateTimeIndex
dataset['Time'] = pd.to_datetime(dataset['Time'],format='%Y-%m-%d %H:%M:%S:%f')
dataset = dataset.set_index('Time')
EgPlot(dataset, 'CWPQL8', 12, 5, 1.5, 2)

#%% New feature plot
dataset = df.copy()
# 将时间戳列转换为 DateTimeIndex
dataset['Time'] = pd.to_datetime(dataset['Time'],format='%Y-%m-%d %H:%M:%S:%f')
dataset = dataset.set_index('Time')

# Convert the units of 11 newly added features
Coef = 40/32767
dataset_convert = feature_unit_convert(dataset, Coef)


# 画单个特征
EgPlot_SingleFeature(dataset_convert,'Door', paras['Lift_Name'], 12, 5, 1.5, 2)
# 画所有特征
EgPlot_NewFeatures(dataset_convert)
#%% vibration plot 

# EgPlot(dataset, 'CWPQL7', 12, 5, 1.5, 2)
EgPlot_RawFeature_Vibration(dataset, 'CWPQL7', 12, 9, 1.5, 2)