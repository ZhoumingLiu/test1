# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 23:27:04 2022

@author: chais
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 01:05:39 2022

@author: chais
"""

from datetime import datetime
import os
import pandas as pd
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from tsmoothie.smoother import *
from sklearn.preprocessing import MinMaxScaler

from utils_lift import *
# from utils_lift_QML3 import *

from paras_lift import paras_QML3

# from utils_OnlineTest import *



#%%
def runMethod(dataset_raw):
 
    dataset_raw['Time'] = pd.to_datetime(dataset_raw['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
    dataset_raw = dataset_raw.set_index('Time')
    # dataset_raw = dataset_raw[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance']]
    dataset = dataset_raw[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance']]
    end_time = dataset.index[-1].strftime('%Y-%m-%d %H:%M:%S.%f')
    # dataset=dataset_raw.copy()
    """读取数据完毕"""
    
    # 加载该电梯的关键参数
    paras = paras_QML3

   
    event_list = []
    CarStat_list = []
    DoorStat_list = []
    
    event_list = RMU_offline(paras, dataset,event_list,end_time)
    
    if len(event_list) == 0:
        
        event_list = LockMode(dataset,paras) # 再判断是不是lock mode
        
        if len(event_list) == 0:
                        
            event_list1 = SafetyTrip(dataset,paras)

            CarSeg_list, DoorSeg_list = Data_segment(dataset, paras, 0.5, 2)
            
            CarStat_list, DoorStat_list, event_list2 = do_action(dataset,paras, CarSeg_list,DoorSeg_list)
            
            event_list = event_list1 + event_list2  
    
            
    if len(event_list) > 0:
        last_status = event_list[-1]['status ID'] # 前端展示事件列表里最后一个事件的状态
    else:
        last_status = 0 # 0 - normal   
        log_text = {
            "time": str(end_time),
            "status ID": 0,
            "event": "Normal",
            "description": "Normal Operation",
            "floor": 'nan',
            "delsign":0              
                }    
        event_list.append(log_text) # 把正常事件存到event_list里   
   
    result, CarSeg_Stats, DoorSeg_Stats = final_output(dataset,paras,last_status,event_list,CarStat_list,DoorStat_list)
  
    return result, CarSeg_Stats, DoorSeg_Stats


#%%
def runMethod_hourly(dataset_raw):
 
    dataset_raw['Time'] = pd.to_datetime(dataset_raw['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S')
    dataset_raw = dataset_raw.set_index('Time')
    dataset = dataset_raw[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance']]

    end_time = dataset.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    # dataset=dataset_raw.copy()
    """读取数据完毕"""
    
    # 加载该电梯的关键参数
    paras = paras_QML3

   
    event_list = []
    # CarStat_list = []
    DoorStat_list = []
    # if dataset.isnull().sum().sum()/(dataset.shape[0]*dataset.shape[1]) > paras['MissingData_Rate']: # nan数据数量大于所取数据数量总数的50%
    if dataset['Brake'].isnull().sum()/dataset.shape[0] > paras['MissingData_Rate']: # nan数据数量大于所取数据数量总数的50%
       
        # last_status = 1 # 1-RMU offline 
        log_text = {
            "time": str(end_time),
            "status ID": 1,
            "event": "RMU offline",
            "description": "Data loss rate > 50%",
            "floor": 'nan',
            "delsign":0              
                            }
        event_list.append(log_text) # 把RMU offline存到event_list里 
        
    else: 

        event_list = LockMode(dataset,paras) # 再判断是不是lock mode
              
    if len(event_list) == 0: # 若既不是offline 也不是lock mode，则进入主程序,首先判断是否是trip，然后do action。
    

        # event_list1 = SafetyTrip(dataset,paras)
        # event_list2 = SafetyInspection(dataset,paras)

        # CarSeg_list, DoorSeg_list = Data_segment(dataset)
        DoorSeg_list = Data_segment_Hourly(dataset, paras, 2)
        
        DoorStat_list, event_list = do_action_Hourly(dataset,paras,DoorSeg_list)
        
        # event_list = event_list1 + event_list2 + event_list3
            
    # if len(event_list) > 0:

    #     last_status = event_list[-1]['status ID'] # 前端展示事件列表里最后一个事件的状态
    # else:
    #     last_status = 0 # 0 - normal   
    #     log_text = {
    #         "time": str(end_time),
    #         "status ID": 0,
    #         "event": "Normal",
    #         "description": "Normal Operation",
    #         "floor": 'nan',
    #         "delsign":0              
    #             }
    #     event_list.append(log_text) # 把正常事件存到event_list里   
   
    result, DoorSeg_Stats = final_output_hourly(dataset,paras,event_list,DoorStat_list)

    return result, DoorSeg_Stats
