# -*- coding: utf-8 -*-
"""
Created on Wed May 25 03:04:34 2022

@author: chais
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 17 00:17:21 2022

@author: chais
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 01:53:33 2022

@author: chais
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 01:42:23 2021

@author: chaisongjian
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 01:02:36 2021

@author: chais
"""
# from datetime import datetime
import os
import pandas as pd
import torch
import numpy as np
from math import ceil

from datetime import datetime

from keras import backend as K
K.clear_session()

from utils_lift import Stats_PerCarSeg
from tsmoothie.smoother import *


#%%


# def Data_segmentV2(dataset, paras):
    
#     dataset = dataset.reset_index()
#     # dataset = dataset.rename(columns={'index': 'Time'})
    
    
#     # use Kalman Filtering to smooth the data 
#     smoother = KalmanSmoother(component='level_trend',
#                           component_noise={'level': 0.1, 'trend': 0.1})    
    
#     dataset.loc[dataset['Distance'] >= 180,'Distance'] = np.nan # remove the outliers
#     dataset.loc[dataset['Distance'] == 0,'Distance'] = np.nan # remove the outliers

#     # dataset['Velocity'] = np.abs(dataset['Distance'].diff() / 0.05)  # calculate the abs velocity
#     # dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  

#     dataset['Velocity'] = np.abs(dataset['Distance'].diff(periods=20))  # calculate the abs velocity
    
#     # 如果brake通道的缺失率在10%以下，才做切分，否则返回空。
#     if dataset['Brake'].isnull().sum()/dataset.shape[0] < 0.1: 
        
#         dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  
        
#         dataset.loc[dataset['Motor'] < 0.15,'Motor'] = 0
#         dataset.loc[dataset['Brake'] < 0.05,'Brake'] = 0
#         dataset.loc[dataset['Door'] < 0.1,'Door'] = 0
#         dataset.loc[dataset['Safety'] < 0,'Safety'] = 0
#         dataset.loc[dataset['Resv-1'] < 0.1,'Resv-1'] = 0
#         dataset.loc[dataset['Resv-2'] < 0.1,'Resv-2'] = 0
#         dataset.loc[dataset['Resv-3'] < 0,'Resv-3'] = 0
#         dataset.loc[dataset['Distance'] < 0,'Distance'] = 0
#         dataset.loc[dataset['Velocity'] < 0,'Velocity'] = 0
        
#         values = dataset[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance','Velocity']]
#         smooth_seq = smoother.smooth(values.T).smooth_data.T # all smoothed sequences, if nan for a whole column, it will be 0.   
#         dataset[['Motor_KF','Brake_KF','Safety_KF','Door_KF','Resv1_KF','Resv2_KF','Resv3_KF','Dist_KF','Vel_KF']] = smooth_seq
        
    
        
#         dataset['Motor_KF'] = dataset['Motor_KF'].apply(lambda x: np.nan if dataset['Motor'].isnull().all() else x)
#         dataset['Brake_KF'] = dataset['Brake_KF'].apply(lambda x: np.nan if dataset['Brake'].isnull().all() else x)
#         dataset['Safety_KF'] = dataset['Safety_KF'].apply(lambda x: np.nan if dataset['Safety'].isnull().all() else x)
#         dataset['Door_KF'] = dataset['Door_KF'].apply(lambda x: np.nan if dataset['Door'].isnull().all() else x)
#         dataset['Resv1_KF'] = dataset['Resv1_KF'].apply(lambda x: np.nan if dataset['Resv-1'].isnull().all() else x)        
#         dataset['Resv2_KF'] = dataset['Resv2_KF'].apply(lambda x: np.nan if dataset['Resv-2'].isnull().all() else x)
#         dataset['Resv3_KF'] = dataset['Resv3_KF'].apply(lambda x: np.nan if dataset['Resv-3'].isnull().all() else x)
#         dataset['Dist_KF'] = dataset['Dist_KF'].apply(lambda x: np.nan if dataset['Distance'].isnull().all() else x)
#         dataset['Vel_KF'] = dataset['Vel_KF'].apply(lambda x: np.nan if dataset['Velocity'].isnull().all() else x)
    
        
#         dataset.loc[dataset['Motor_KF'] < 0,'Motor_KF'] = 0
#         dataset.loc[dataset['Brake_KF'] < 0.15,'Brake_KF'] = 0
#         dataset.loc[dataset['Safety_KF'] < 0,'Safety_KF'] = 0
#         dataset.loc[dataset['Door_KF'] < 0,'Door_KF'] = 0
#         dataset.loc[dataset['Resv1_KF'] < 0,'Resv1_KF'] = 0
#         dataset.loc[dataset['Resv2_KF'] < 0,'Resv2_KF'] = 0
#         dataset.loc[dataset['Resv3_KF'] < 0,'Resv3_KF'] = 0
#         dataset.loc[dataset['Dist_KF'] < 0,'Dist_KF'] = 0
#         dataset.loc[dataset['Vel_KF'] < 0.01,'Vel_KF'] = 0
    
    
    
    
            
#         CarSeg_list = []
#         DoorSeg_list = []
        
#         dataset['carseg_flag'] = np.sign(dataset['Brake_KF']-0.01) # -1 indicates brake close and 1 indicates brake open    
#         dataset['doorseg_flag'] = np.sign(paras['thres_resv3']-dataset['Resv-3']) # series with -1 and 1, -1 indicates door close and 1 indicates door open
        
#         carseg_group = dataset[dataset['carseg_flag'] == 1].groupby((dataset['carseg_flag'] != 1).cumsum())
#         doorseg_group = dataset[dataset['doorseg_flag'] == 1].groupby((dataset['doorseg_flag'] != 1).cumsum())
        
#         for k, v in carseg_group:
            
#             CarSeg_list.append(v)  # Car motion cycle list
    
#         for k, v in doorseg_group:
            
#             DoorSeg_list.append(v)  # Door motion cycle list
    
    
#         # 删除第一个不完整的car cycle
#         if len(CarSeg_list)>0:
#             if CarSeg_list[0].iloc[0]['Brake_KF']>0.5:
        
#                 del (CarSeg_list[0])
                
#         # 删除最后一个不完整的car cycle
                
#         if len(CarSeg_list)>0:            
#             if CarSeg_list[-1].iloc[-1]['Brake_KF']>0.5:
        
#                 del (CarSeg_list[-1])
    
                
#         # 删除car segment中小于3s的cycle        
#         CarSeg_list = list(filter(lambda x: len(x)>60, CarSeg_list))
#         # 删除door segment中为空的cycle        
#         DoorSeg_list = list(filter(lambda x: len(x)>0, DoorSeg_list))
        
#         # 删除door segment中小于5s的cycle        
#         DoorSeg_list = list(filter(lambda x: len(x)>100, DoorSeg_list))
        
#         # 删除door segment中首尾segment     
#         if len(DoorSeg_list)>0:
         
#             if (DoorSeg_list[0].iloc[0]['Time'] == dataset['Time'].iloc[0]) & (DoorSeg_list[0].iloc[-1]['Time'] != dataset['Time'].iloc[-1]):
#                 del (DoorSeg_list[0])
                     
#         if len(DoorSeg_list)>0:
      
#             if (DoorSeg_list[-1].iloc[-1]['Time'] == dataset['Time'].iloc[-1]) & (len(DoorSeg_list[-1])<400): # 末尾小于20s的删除
#                 del (DoorSeg_list[-1])            
    
    
        
#         # DoorSeg_list = list(filter(lambda x: len(x)>140, DoorSeg_list))
        
#     else:
#         CarSeg_list = []
#         DoorSeg_list = []
            
    
#     return CarSeg_list, DoorSeg_list


def Stats_PerDoorSeg_QML3(seq, paras, event_list): # Statistic information for each car segment/cycle
    ############################################################
    ################  1. Traffic Information ###################
    ############################################################
    start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
    hour = seq.iloc[-1]['Time'].hour                    

    ############################################################
    ################  2. Key Parameters Calculation ############
    ############################################################   
    distance = np.abs(seq.iloc[-1]['Distance']-seq.iloc[0]['Distance'])
    if not pd.isnull(paras['line_Door']):
        num_Door = len(np.argwhere(np.diff(np.sign(paras['line_Door']-seq['Door_KF']))))/2 # number of door closing and opening pairs
        num_Door = ceil(num_Door)
    else:
        num_Door = np.nan
        
    if num_Door > 0:
        DoorI_peak = seq['Door_KF'].max()
    else:
        DoorI_peak = np.nan
        
        
    if pd.isnull(paras['thres_resv3']):
        seq_dooropen = seq.loc[(seq['Safety']>=paras['Dooropen_SafetyI_range'][0]) & (seq['Safety']<=paras['Dooropen_SafetyI_range'][1])]
        DoorOpen_Duration = len(seq_dooropen)/20
    else:
        DoorOpen_Duration = len(seq.loc[seq['Resv-3'] < paras['thres_resv3']])/20 # door open duration within each cycle    
    ############################################################
    ############  3. Event Detection based on Rules ############
    ###################################s########################
    
    # UCM     
    if distance > 1000.3:
        log_text = {
            "time": str(end_time),
            "status ID": 2.1,
            "event": "UCM",
            "description": "lift car has travelled beyond door zone (+/- 300 mm of floor level)",
                }
        event_list.append(log_text)     

    # if DoorOpen_Duration > 30 and duration < 55: # the lift is very frequently operated as "Staff Mode", which the door will keep open if the lift is idle. 
    if paras['DoorOpenLong_FLAG'] == 1 and 30 < DoorOpen_Duration < 55 and Stop_F != 0: 

        log_text = {
            "time": str(end_time),
            "status ID": 3.1,
            "event": "Door anomaly",
            "description": "Door opens for too long time",
                }
        event_list.append(log_text)   
        
        
    if paras['AI_FLAG'] != 1: # when AI is not adopted
    
        if num_Door >= paras['thres_numDoor']:
            
            log_text = {
                "time": str(end_time),
                "status ID": 3.1,
                "event": "Door anomaly",
                "description": "Excessive door open & close actions",
                    }
            event_list.append(log_text)    

        if DoorI_peak < paras['DrIpeak_Range'][0] or DoorI_peak > paras['DrIpeak_Range'][1]:
            
            log_text = {
                "time": str(end_time),
                "status ID": 3.1,
                "event": "Door anomaly",
                "description": "Anomaly door motor current magnitude:" + str(round(DoorI_peak,2))
                    }
            event_list.append(log_text) 
            
    if np.isnan(paras['Floor']).all():        
        Stop_F=np.nan
    else:
        Stop_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor'])) - 1 # STOP floor of this cycle

            
    DoorStat_text = {
        'start_time':start_time,        
        'end_time':end_time,
        'duration':duration,        
        'hour':hour,
        'DoorI_peak':round(DoorI_peak,2),
        'num_Door':num_Door,
        'DoorOpen_Duration':round(DoorOpen_Duration,2),
        'Stop_F': Stop_F
        } 
    
    DoorStat_text = {k: str(DoorStat_text[k]) if pd.isnull(DoorStat_text[k]) else DoorStat_text[k] for k in DoorStat_text }

    return DoorStat_text, event_list

def do_action_QML3(dataset, paras, CarSeg_list, DoorSeg_list):
    event_list = []
    CarStat_list = []
    DoorStat_list = []
       
    AI_FLAG = paras['AI_FLAG']
    # Floor = paras['Floor']

    ##计算loss
    # criterion = torch.nn.MSELoss(reduction='sum')
    # criterion2 = torch.nn.L1Loss(reduction='sum')
    ############################################################
    ################  1. CarSeg_list ########################
    ############################################################
    if len(CarSeg_list) > 0:
        
        # brake_list = []
        # brake_result = []
        # motor_list = []
        # motor_result = []
        for i in range(len(CarSeg_list)):
            seq = CarSeg_list[i]
            # end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            
            ####################### Rules #########################
            CarStat_text,event_list = Stats_PerCarSeg(seq,paras,event_list)


            CarStat_list.append(CarStat_text) 


    

    ############################################################
    ################  2. DoorSeg_list ########################
    ############################################################
    if len(DoorSeg_list) > 0:

        # door_list = []
        # door_result = []
        for i in range(len(DoorSeg_list)):
            seq = DoorSeg_list[i]        
            # end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
             ####################### Rules #########################
            
            DoorStat_text,event_list = Stats_PerDoorSeg_QML3(seq,paras,event_list)
    
            DoorStat_list.append(DoorStat_text) 
            

            

    return CarStat_list, DoorStat_list, event_list






