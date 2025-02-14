#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:41:40 2022

@author: chaisongjian

"""

import numpy as np

def velocity_cal(dataset):
    dataset['vel'] = dataset['Distance'].diff(periods=20)  # calculate the  velocity
    dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  
    return dataset

def LPF(dataset,k):
    i = 0
    dataset['vel_filter'] = dataset['vel'][0]
    N = len(dataset['vel'])
    while i < (N-1):
        dataset['vel_filter'][i+1] = k*dataset['vel'][i+1] + (1-k)*dataset['vel_filter'][i] 
        
        i = i+1
        
    return dataset

def LPF_Dist(dataset,k):
    i = 0
    dataset['Dist_LPF'] = dataset['Distance'][0]
    N = len(dataset['Distance'])
    while i < (N-1):
        dataset['Dist_LPF'][i+1] = k*dataset['Distance'][i+1] + (1-k)*dataset['Dist_LPF'][i] 
        
        i = i+1
        
    return dataset

def acc_cal(dataset): # 加速度计算
    dataset['vel'] = dataset['Distance'].diff(periods=20)  # calculate the  velocity
    dataset['acc'] = dataset['vel'].diff(periods=1)  # calculate the  velocity

    dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  
    return dataset

def delta_p(dataset): # 计算位移
    dataset['vel'] = dataset['Distance'].diff(periods=20)  # calculate the  velocity
    delta_t = 0.5
    a_max = 10
    dataset['delta_Pcal'] = dataset['Distance'].diff(periods=10)
    dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  

    dataset['delta_Pmax'] = dataset['vel']*delta_t + 0.5*a_max*delta_t*delta_t # 位移
    dataset['delta_Pmin'] = dataset['vel']*delta_t - 0.5*a_max*delta_t*delta_t # calculate the  velocity
    
    
    return dataset

def rmv_DistSpike(dataset): # 至少取当前时刻前2s的数
    
    delta_t = 0.5
    a_max = 20
    N = len(dataset['Distance'])
    i = 21
    flag = 0
    k = 0
    Index_start=[]
    Index_end=[]
    dataset['Dist_raw'] = dataset['Distance'].copy()
    while i < (N-1):
        vel = dataset['Distance'][i] - dataset['Distance'][i-20] 
        dPcal = dataset['Distance'][i] - dataset['Distance'][i-10]
        dP_max = vel*delta_t + 0.5*a_max*delta_t*delta_t 
        dP_min = vel*delta_t - 0.5*a_max*delta_t*delta_t 
        
        if not dP_min < dPcal < dP_max:
            vel = dataset['Distance'][i-1] - dataset['Distance'][i-21] 
            dataset['Distance'][i]=dataset['Distance'][i-1]+vel*delta_t*0.1
            
            if flag == 0:
                k  += 1
                
                # dataset['Index_start'][k] = i
                Index_start.append(i)
                flag = 1   
            
        else:
            if flag == 1:
            
                
                # dataset['Index_end'][k] = i
                Index_end.append(i)

                flag = 0  
                k  += 1

        
        i = i+1
                        
            
    if len(Index_start)>0:
        for i,v in enumerate(Index_start):
            dataset['Distance'].iloc[(Index_start[i]-10):(Index_end[i]+10)]=np.nan        
            #dataset.loc['Distance',(index_start-1):(index_end+1)] = np.nan
    dataset['Distance'] = dataset['Distance'].interpolate()
        
    return dataset,Index_start,Index_end



def get_median_filtered(signal, threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal




dataset = velocity_cal(dataset)

dataset = LPF(dataset,0.1)

seq_filter = LPF_Dist(seq,0.1)

seq_acc = acc_cal(seq)

seq_deltaP = delta_p(seq)

seq_deltaP[['delta_Pcal','delta_Pmax','delta_Pmin']].plot()


seq,Index_start,Index_end = rmv_DistSpike(seq)


seq = get_median_filtered(seq['Distance'], threshold=3)



from pandas import Rolling

# threshold = 3
# df['pandas'] = rolling_median(df['u'], window=3, center=True).fillna(method='bfill').fillna(method='ffill')


# seq['Dist_filter'] = seq['Distance'].rolling(window=3).median().fillna(method='bfill').fillna(method='ffill')

# seq['median']= seq['Distance'].rolling(window=6).median()
# seq['std'] = seq['Distance'].rolling(window=6).std()

# #filter setup
# aa = seq[(seq.Distance <= seq['median']+3*seq['std']) & (seq.Distance >= seq['median']-3*seq['std'])]


import scipy.signal as signal
aa = signal.medfilt(seq['Distance'],9)

