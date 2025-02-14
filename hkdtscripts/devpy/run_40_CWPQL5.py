# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:21:28 2024

@author: chais
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:44:42 2024

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
from paras_lift import paras_CWPQL5



#%%
def runMethod(dataset_raw):
 
    dataset_raw['Time'] = pd.to_datetime(dataset_raw['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
    dataset_raw = dataset_raw.set_index('Time')
    # dataset_raw = dataset_raw[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance']]
    dataset = dataset_raw[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance','floor',
                           'velocity','mileage','openCloseDoorNum','door','workMode','motion',
                           'cumulativeRunNum']]
    end_time = dataset.index[-1].strftime('%Y-%m-%d %H:%M:%S.%f')
    # dataset=dataset_raw.copy()
    """读取数据完毕"""
    
    # 加载该电梯的关键参数
    paras = paras_CWPQL5

   
    event_list = []
    CarStat_list = []
    DoorStat_list = []
    
    event_list = RMU_offline(paras, dataset,event_list,end_time)
    
    if len(event_list) == 0:
        
        event_list = LockMode(dataset,paras) # 再判断是不是lock mode
        
        if len(event_list) == 0:
            
            mask = dataset['Brake'] < 0.25
            dataset.loc[mask,'Brake'] = 0
    
            
            event_list1 = SafetyTrip(dataset,paras)

            CarSeg_list, DoorSeg_list = Data_segment(dataset, paras, 0.5, 1)
            
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


