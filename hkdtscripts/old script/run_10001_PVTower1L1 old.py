# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:39:40 2021

@author: chais
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:08:25 2021

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


#%% Define main function

def runMethod(dataset_raw):
 
    dataset_raw['Time'] = pd.to_datetime(dataset_raw['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
    dataset_raw = dataset_raw.set_index('Time')
    # dataset_raw = dataset_raw[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance']]
    dataset = dataset_raw[['Motor','Brake','Safety','Door','Resv-3','Distance']]
    end_time = dataset.index[-1]
    # dataset=dataset_raw.copy()
    """读取数据完毕"""
    
    # 定义该电梯的关键参数
    paras = {
        "MissingData_Rate": 0.5,
        "Locked_SafetyI_Max": np.nan,
        "Locked_SafetyI_Min": np.nan, 
        "Tripped_SafetyI_Max": np.nan,
        "Tripped_SafetyI_Min": np.nan,
        "BrIsteady_Max": 2,
        "BrIsteady_Min": 0.75,
        "BrIpeak_Max": 4.2,
        "BrIpeak_Min": 1.0,
        "MotIpeak_Max": 60,
        "MotIpeak_Min": 10,
        "MotIsteady_Max": np.nan,
        "MotIsteady_Min": np.nan,
        "thres_numDoor": 6,
        "DrIpeak_Max": np.nan,
        "line_Door": 0.15,
        "MotZero_Rate": 0.9,
        "RatedSpeed": 2.0,                        
        "DoorError_max": np.nan,
        "MotorError_max": np.nan,
        "BrakeError_max": np.nan
        
        }
   
    event_list = []

    if dataset.isnull().sum().sum()/(dataset.shape[0]*dataset.shape[1]) > paras['MissingData_Rate']: # nan数据数量大于所取数据数量总数的10%
        
        # last_status = 1 # 1-RMU offline 
        log_text = {
            "time": str(end_time),
            "status ID": 1,
            "event": "RMU offline",
            "description": "Data loss rate > 50%",
                            }
        event_list.append(log_text) # 把RMU offline存到event_list里 
        
    else: 

        dataset = dataset.interpolate() # 用插值法补全缺失数据 
        dataset.loc[dataset['Brake'] < 0.25,'Brake'] = 0
        dataset.loc[dataset['Door'] < 0.1,'Door'] = 0
        
        # 先判断safety 看是不是3.3-Locked（锁机）模式 
        # if (dataset.iloc[-100: ]['Safety'].median() >= paras['Locked_SafetyI_Min']) & (dataset.iloc[-100: ]['Safety'].median() <= paras['Locked_SafetyI_Max']): 
                           
        #     # last_status = 3.3  # 3-out of service -> 3.3 Locked
        #     log_text = {
        #         "time": str(end_time),
        #         "status ID": 3.3,
        #         "event": "Locked",
        #         "description": "Out of service (locked)",
        #                         }
        #     event_list.append(log_text) # 把锁机事件存到event_list里        
        
        # elif (dataset.iloc[-100: ]['Safety'].median() >= paras['Tripped_SafetyI_Min']) & (dataset.iloc[-100: ]['Safety'].median() <=  paras['Tripped_SafetyI_Max']):
            
        #     # last_status = 3.2   # 3-out of service -> 3.2 Safety tripped
        #     log_text = {
        #         "time": str(end_time),
        #         "status ID": 3.2,
        #         "event": "Safety tripped",
        #         "description": "Out of service (safety tripped)",
        #             }
        #     event_list.append(log_text) # 把锁机事件存到event_list里   

            
        # else:
        
        CarSeg_list, DoorSeg_list = Data_segment(dataset)
        event_list = do_action_RULES(dataset,paras, CarSeg_list,DoorSeg_list)
            
    if len(event_list) > 0:
        last_status = event_list[-1]['status ID'] # 前端展示事件列表里最后一个事件的状态
    else:
        last_status = 0 # 0 - normal   
        log_text = {
            "time": str(end_time),
            "status ID": 0,
            "event": "Normal",
            "description": "Normal Operation",
                }
        event_list.append(log_text) # 把正常事件存到event_list里   

        
    result = {
        "Lift_ID": 10001,
        "Lift_Name": "PVTower1L1",
        "last_status": last_status,
        "event_list": event_list,
        "period": {
            "start": dataset.index[0],
            "end": dataset.index[-1],
                },
        "post_time":pd.to_datetime(datetime.now())
            }
    
    return result



