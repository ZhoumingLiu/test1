# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 01:05:39 2021

@author: chais
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 01:42:48 2021

@author: chaisongjian
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 01:16:02 2021

@author: chaisongjian
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 01:11:29 2021

@author: chaisongjian
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






#%%
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
        "Locked_SafetyI_Max": 0.11,
        "Locked_SafetyI_Min": 0.09, 
        "Tripped_SafetyI_Max": 0.151,
        "Tripped_SafetyI_Min": 0.111,
        "Voltage_Dip_Max": 0.09,
        "Voltage_Dip_Min": 0.0,        
        "InspectionRun_Max": -1,
        "InspectionRun_Min": -2,
        "InspectionStop_Max": -3,
        "InspectionStop_Min": -4, 
        "BrIsteady_Max": np.nan,
        "BrIsteady_Min": np.nan,
        "BrIpeak_Max": np.nan,
        "BrIpeak_Min": np.nan,
        "MotIpeak_Max": np.nan,
        "MotIpeak_Min": np.nan,
        "MotIsteady_Max": np.nan,
        "MotIsteady_Min": np.nan,
        "thres_numDoor": np.nan,
        "DrIpeak_Max": np.nan,
        "RatedSpeed": 1.6,        
        "DoorError_max": 25,
        "MotorError_max": 10,
        "BrakeError_max": 15
        
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

        # dataset = dataset.interpolate() # 用插值法补全缺失数据 
        # dataset.loc[dataset['Brake'] < 0.25,'Brake'] = 0
        # dataset.loc[dataset['Door'] < 0.1,'Door'] = 0
        
        event_list = do_action_SafetyCircuit(dataset,paras) # 把safetycircuit related events存到event_list里 
              
    if len(event_list) == 0:
    
        CarSeg_list, DoorSeg_list = Data_segment(dataset)
        event_list = do_action_LSTMAE(dataset,paras, CarSeg_list,DoorSeg_list)

            
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
        "Lift_ID": 10020,
        "Lift_Name": "EMSDHQL12",
        "last_status": last_status,
        "event_list": event_list,
        "period": {
            "start": dataset.index[0],
            "end": dataset.index[-1],
                },
        "post_time":pd.to_datetime(datetime.now())
            }
    
    return result


