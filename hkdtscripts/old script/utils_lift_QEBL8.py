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
from tsmoothie.smoother import *


#%%


# def Data_segmentV2(dataset, paras):
    
#     dataset = dataset.reset_index()
#     # dataset = dataset.rename(columns={'index': 'Time'})
    
    
#     # use Kalman Filtering to smooth the data 
#     smoother = KalmanSmoother(component='level_trend',
#                           component_noise={'level': 0.1, 'trend': 0.1})    
    
#     dataset.loc[dataset['Distance'] >= 100,'Distance'] = np.nan # remove the outliers
#     dataset.loc[dataset['Distance'] == 0,'Distance'] = np.nan # remove the outliers

#     # dataset['Velocity'] = np.abs(dataset['Distance'].diff() / 0.05)  # calculate the abs velocity
#     # dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  

#     dataset['Velocity'] = np.abs(dataset['Distance'].diff(periods=20))  # calculate the abs velocity

#     # 如果brake通道的缺失率在10%以下，才做切分，否则返回空。
#     if dataset['Brake'].isnull().sum()/dataset.shape[0] < 0.1:   
        
#         dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  
        
#         # dataset.loc[dataset['Motor'] < 0.15,'Motor'] = 0
#         dataset.loc[dataset['Brake'] < 0.15,'Brake'] = 0
#         dataset.loc[dataset['Door'] < 0.1,'Door'] = 0
#         dataset.loc[dataset['Safety'] < 0,'Safety'] = 0
#         # dataset.loc[dataset['Resv-1'] < 0.1,'Resv-1'] = 0
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
    
        
#         # dataset.loc[dataset['Motor_KF'] < 0,'Motor_KF'] = 0
#         dataset.loc[dataset['Brake_KF'] < 0.15,'Brake_KF'] = 0
#         dataset.loc[dataset['Safety_KF'] < 0,'Safety_KF'] = 0
#         dataset.loc[dataset['Door_KF'] < 0,'Door_KF'] = 0
#         # dataset.loc[dataset['Resv1_KF'] < 0,'Resv1_KF'] = 0
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



# def Stats_PerCarSeg_DC(seq, paras, event_list): # Statistic information for each car segment/cycle

#     ############################################################
#     ################  1. Traffic Information ###################
#     ############################################################
#     start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
#     end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
#     duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
#     hour = seq.iloc[-1]['Time'].hour                 
#     # mileage = np.abs(seq['Distance'].iloc[-1] - seq['Distance'].iloc[0])

#     mileage = np.abs(seq['Distance'].iloc[-10:].median() - seq['Distance'].iloc[:10].median())
    
#     if np.isnan(paras['Floor']).all():        
#         Depart_F = Arrive_F = Fs_travelled = F_Travelled = Dir_Travelled = position_dev = np.nan
#     else:
#         Depart_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor'])) - 1 # start floor of this cycle
#         Arrive_F = np.nanargmin(np.abs(seq['Distance'].iloc[-1] - paras['Floor'])) - 1 # end floor of this cycle        
#         # position_dev = np.nanmin(np.abs(seq['Distance'].iloc[-1] - paras['Floor'])) # stop position deviation from the desired floor
#         position_dev = np.nanmin(np.abs(seq['Distance'].iloc[-10:].median() - paras['Floor'])) # stop position deviation from the desired floor
#         Fs_travelled = Arrive_F - Depart_F # the floors the elevator travels, + upward, - downward
#         F_Travelled = np.abs(Fs_travelled) # the floors the elevator travels
#         Dir_Travelled = np.sign(Fs_travelled) # +1 upward, -1 downward, 0 releveling
    
#     ############################################################
#     ################  2. Key Parameters Calculation ############
#     ############################################################
    
#     Resv1I_peak = seq['Resv1_KF'].abs().max()  # Armature peak current


#     if len(seq) >= 240: # trip 大于12s

#         MotorI_start = seq['Motor_KF'].iloc[:10].mean()  # Full Field current
#         MotorI_peak = seq['Motor_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # Run Field current

#         Resv1I_start = seq['Resv1_KF'].iloc[40:70].mean()  # Armature starting current
#         Resv1I_brake = seq['Resv1_KF'].iloc[-50:-25].mean()  # Armature braking current
#         Resv1I_steady = seq['Resv1_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # Armature steady current
        
#         BrakeI_min = seq['Brake'].iloc[30:-30].min() # minimum brake current when car runs 


#     elif 80 < len(seq) < 240: # trip在 4s-12s之间

#         MotorI_start = seq['Motor_KF'].iloc[:10].mean()  # Full Field current
#         MotorI_peak = np.nan  # Run Field current

        
#         Resv1I_start = seq['Resv1_KF'].iloc[40:70].mean()  # Armature starting current
#         Resv1I_brake = seq['Resv1_KF'].iloc[-50:-25].mean()  # Armature braking current
#         Resv1I_steady = np.nan  # Armature steady current
        
#         BrakeI_min = seq['Brake'].iloc[30:-30].min() # minimum brake current when car runs 


        
#     else:
        
#         MotorI_start = np.nan  # Full Field current
#         MotorI_peak = np.nan  # Run Field current

        
#         Resv1I_start = np.nan  # Armature starting current
#         Resv1I_brake = np.nan  # Armature braking current
#         Resv1I_steady = np.nan  # Armature steady current
        
#         BrakeI_min = seq['Brake'].iloc[5:-5].min() # minimum brake current when car runs 
      
        
#     BrakeI_steady = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current
#     BrakeI_peak = seq['Brake_KF'].max() # peak brake current when car runs 
    
#     SafetyI_run = seq['Safety'].mean() #average safety current when car runs
#     SafetyI_max = seq['Safety'].max() #max safety current when car runs
    

#     Resv3I_run = seq['Resv-3'].mean() # average resv3 current when car runs
        
#     Speed_peak = seq['Vel_KF'].max() # peak speed  when car runs   
    
    
    
        
#     ############################################################
#     ############  3. Event Detection based on Rules ############
#     ############################################################
#     # Motor field current anomaly      
#     if not paras['RunField_Range'][0] < MotorI_peak < paras['RunField_Range'][1]: # out of Run field current
#          log_text = {
#              "time": str(end_time),
#              "status ID": 3.3,
#              "event": "Motor anomaly",
#              "description": "anomaly run field current magnitude:" + str(round(MotorI_peak,2)),
#                  }
#          event_list.append(log_text) 

#     if not paras['FullField_Range'][0] < MotorI_start < paras['FullField_Range'][1]: # out of Full field current
#          log_text = {
#              "time": str(end_time),
#              "status ID": 3.3,
#              "event": "Motor anomaly",
#              "description": "anomaly full field current magnitude:" + str(round(MotorI_start,2)),
#                  }
#          event_list.append(log_text) 
         
         
#     # Motor armature current anomaly 
#     if not paras['ArmaturePeak_Range'][0] < Resv1I_peak < paras['ArmaturePeak_Range'][1]: # out of Armature Peak current
#          log_text = {
#              "time": str(end_time),
#              "status ID": 3.3,
#              "event": "Motor anomaly",
#              "description": "anomaly armature peak current magnitude:" + str(round(Resv1I_peak,2)),
#                  }
#          event_list.append(log_text) 

#     if not paras['ArmatureStart_Range'][0] < Resv1I_start < paras['ArmatureStart_Range'][1]: # out of Armature Starting current
#          log_text = {
#              "time": str(end_time),
#              "status ID": 3.3,
#              "event": "Motor anomaly",
#              "description": "anomaly armature starting current magnitude:" + str(round(Resv1I_start,2)),
#                  }
#          event_list.append(log_text) 

#     if not paras['ArmatureBrake_Range'][0] < Resv1I_brake < paras['ArmatureBrake_Range'][1]: # out of Armature Braking current
#          log_text = {
#              "time": str(end_time),
#              "status ID": 3.3,
#              "event": "Motor anomaly",
#              "description": "anomaly armature braking current magnitude:" + str(round(Resv1I_brake,2)),
#                  }
#          event_list.append(log_text) 

#     if not paras['ArmatureSteady_Range'][0] < Resv1I_steady < paras['ArmatureSteady_Range'][1]: # out of Armature Steady current
#          log_text = {
#              "time": str(end_time),
#              "status ID": 3.3,
#              "event": "Motor anomaly",
#              "description": "anomaly armature steady current magnitude:" + str(round(Resv1I_steady,2)),
#                  }
#          event_list.append(log_text)
         
#     # Brake Faults         # out of the normal brake current range
#     if not paras['BrIsteady_Range'][0] < BrakeI_steady < paras['BrIsteady_Range'][1] and ~np.isnan(BrakeI_steady): # out of the normal brake current range
#          log_text = {
#              "time": str(end_time),
#              "status ID": 2.3,
#              "event": "Brake Faults",
#               "description": "anomaly brake steady current magnitude:" + str(round(BrakeI_steady,2)),
#                  }
#          event_list.append(log_text)

#     # Brake Faults          # out of the normal brake current range
#     if not paras['BrIpeak_Range'][0] < BrakeI_peak < paras['BrIpeak_Range'][1]: # out of the normal brake current range
#          log_text = {
#              "time": str(end_time),
#              "status ID": 2.3,
#              "event": "Brake Faults",
#              "description": "anomaly brake peak current magnitude:" + str(round(BrakeI_peak,2)),
#                  }
#          event_list.append(log_text)       

#         # Brake Unsmooth Operation          #  Brake ramp down, safety surge up and motor rise a little 
#     if (BrakeI_min < paras['BrIsteady_Range'][0]) & (SafetyI_max > 2.5*paras["Run_SafetyI_range"][1]):
#         log_text = {
#             "time": str(end_time),
#             "status ID": 3.2,
#             "event": "Brake unsmooth operation",
#             "description": "Brake unsmooth operation",
#                 }
#         event_list.append(log_text) # 把brake异常存到event_list里        
        
#     # ACO     
#     if Dir_Travelled == 1:  # 判断电梯是否处于上升状态

#         seq_overspeed = seq.loc[seq['Velocity'] > 1.15 * paras['RatedSpeed']] # When lift car is travelling upward and >115% of rated speed

#         if len(seq_overspeed) > 80: # overspeed 超过4s
#             log_text = {
#                 "time": str(end_time),
#                 "status ID": 2.2,
#                 "event": "ACO",
#                  "description": "Lift ascending over speed ",
#                     }
#             event_list.append(log_text)    
            
#     # Inspection     
#     if 0.1 < Speed_peak < 0.5 and ~np.isnan(Arrive_F):  # 判断是否为慢车检修模式
#         log_text = {
#             "time": str(end_time),
#             "status ID": 4,
#             "event": "Inspection mode",                    
#             "description": "Inspection mode - maintenance",
#                 }
#         event_list.append(log_text)    


#     # Stop at Non-service Floor (原来叫作 Over Travel)     
#     if 0.3 < position_dev < 7 and seq['Velocity'].iloc[-1] < 0.5 and seq['Motor_KF'].iloc[-1] < 10:
#         log_text = {
#             "time": str(end_time),
#             "status ID": 2.7,
#             "event": "Stop at Non-service Floor",                    
#             "description": "Lift stops at the level not at the proper position (>+/- 300mm)"
#                 }
#         event_list.append(log_text)  

#     # Sudden Stop     
#     # if 0.5 < seq['Velocity'].iloc[-10:].median() < 3 and seq['Motor_KF'].iloc[-1] > 10:  #用该trip的最后1s速度的中位数判断以及最后一个Motor Current值判断    
#     #用该trip的最后0.5s速度的中位数或者距离偏移 判断以及最后一个Motor Current值判断   
#     if seq['Motor_KF'].iloc[-1] > 10 and (0.5 < seq['Velocity'].iloc[-10:].median() < 3 or 0.3 < position_dev < 7):        
#         log_text = {
#             "time": str(end_time),
#             "status ID": 2.8,
#             "event": "Sudden stop",                    
#             "description": "Lift suddenly stops at the mid of travelling "
#                 }
#         event_list.append(log_text)  

#     # Start failure     
#     if mileage < 0.2 and ~np.isnan(Arrive_F) and Resv3I_run > paras['thres_resv3']: # (1)没移动；（2）Arrive_F 有值; (3) Resv3 是处于电梯运行的level
#     # if seq['Distance'].diff().max()<0.01 and ~np.isnan(Arrive_F) and Resv3I_run > paras['thres_resv3']: # (1)没有任何位移变化（无法判断）；（2）Arrive_F 有值; (3) Resv3 是处于电梯运行的level
    
#         log_text = {
#             "time": str(end_time),
#             "status ID": 2.9,
#             "event": "Start failure",                    
#             "description": "Lift cannot start successfully ",
#                 }
#         event_list.append(log_text)  


         
#     CarStat_text = {
#         'start_time':start_time,
#         'end_time':end_time,
#         'duration':duration,
#         'hour':hour,
#         'mileage':round(mileage,2),
#         'Depart_F':Depart_F,
#         'Arrive_F':Arrive_F,
#         'Fs_travelled':Fs_travelled,
#         'F_Travelled':F_Travelled,
#         'Dir_Travelled':Dir_Travelled,
#         'MotorI_start':round(MotorI_start,2), # Full Field current
#         'MotorI_peak':round(MotorI_peak,2),   # Run Field current
#         'MotorI_brake':np.nan, 
#         'MotorI_steady':np.nan,         
#         'BrakeI_steady':round(BrakeI_steady,2),
#         'BrakeI_peak':round(BrakeI_peak,2),
#         'SafetyI_run':round(SafetyI_run,2),
#         'Resv1I_start':round(Resv1I_start,2), #Armature Starting current
#         'Resv1I_peak':round(Resv1I_peak,2), #Armature Peak current
#         'Resv1I_brake':round(Resv1I_brake,2),  #Armature Brake current
#         'Resv1I_steady':round(Resv1I_steady,2), #Armature Steady current          
#         'Resv2I_start':np.nan,
#         'Resv2I_peak':np.nan,
#         'Resv2I_brake':np.nan, 
#         'Resv2I_steady':np.nan,        
#         'Resv3I_run':round(Resv3I_run,2),
#         'Speed_peak':round(Speed_peak,2)
#         } 
    
#     CarStat_text = {k: str(CarStat_text[k]) if pd.isnull(CarStat_text[k]) else CarStat_text[k] for k in CarStat_text }
            
#     return CarStat_text, event_list


# def Stats_PerDoorSeg_QEBL8(seq, paras, event_list): # Statistic information for each car segment/cycle
#     ############################################################
#     ################  1. Traffic Information ###################
#     ############################################################
#     start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
#     end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
#     duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
#     hour = seq.iloc[-1]['Time'].hour                    

#     ############################################################
#     ################  2. Key Parameters Calculation ############
#     ############################################################   
#     distance = np.abs(seq.iloc[-1]['Distance']-seq.iloc[0]['Distance'])
#     if not pd.isnull(paras['line_Door']):
#         num_Door = len(np.argwhere(np.diff(np.sign(paras['line_Door']-seq['Door_KF']))))/4 # number of door closing and opening pairs
#         num_Door = ceil(num_Door)
#     else:
#         num_Door = np.nan
        
#     if num_Door > 0:
#         DoorI_peak = seq['Door_KF'].max()
#     else:
#         DoorI_peak = np.nan
        
        
#     if pd.isnull(paras['thres_resv3']):
#         seq_dooropen = seq.loc[(seq['Safety']>=paras['Dooropen_SafetyI_range'][0]) & (seq['Safety']<=paras['Dooropen_SafetyI_range'][1])]
#         DoorOpen_Duration = len(seq_dooropen)/20
#     else:
#         DoorOpen_Duration = len(seq.loc[seq['Resv-3'] < paras['thres_resv3']])/20 # door open duration within each cycle    
#     ############################################################
#     ############  3. Event Detection based on Rules ############
#     ###################################s########################
    
#     # UCM     
#     if distance > 1000.3:
#         log_text = {
#             "time": str(end_time),
#             "status ID": 2.1,
#             "event": "UCM",
#             "description": "lift car has travelled beyond door zone (+/- 300 mm of floor level)",
#                 }
#         event_list.append(log_text)     

#     # if DoorOpen_Duration > 30 and duration < 55: # the lift is very frequently operated as "Staff Mode", which the door will keep open if the lift is idle. 
#     if paras['DoorOpenLong_FLAG'] == 1 and 30 < DoorOpen_Duration < 55 and Stop_F != 0: 

#         log_text = {
#             "time": str(end_time),
#             "status ID": 3.1,
#             "event": "Door anomaly",
#             "description": "Door opens for too long time",
#                 }
#         event_list.append(log_text)   
        
        
#     if paras['AI_FLAG'] != 1: # when AI is not adopted
    
#         if num_Door >= paras['thres_numDoor']:
            
#             log_text = {
#                 "time": str(end_time),
#                 "status ID": 3.1,
#                 "event": "Door anomaly",
#                 "description": "Excessive door open & close actions",
#                     }
#             event_list.append(log_text)    

#         if DoorI_peak < paras['DrIpeak_Range'][0] or DoorI_peak > paras['DrIpeak_Range'][1]:
            
#             log_text = {
#                 "time": str(end_time),
#                 "status ID": 3.1,
#                 "event": "Door anomaly",
#                 "description": "Anomaly door motor current magnitude",
#                     }
#             event_list.append(log_text) 
            
#     if np.isnan(paras['Floor']).all():        
#         Stop_F=np.nan
#     else:
#         Stop_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor'])) - 1 # STOP floor of this cycle

#     #### 检测是否有releveling case 发生 ####           
#     Relvl_list = []

#     seq['carseg_flag'] = np.sign(seq['Brake_KF']-0.01) # -1 indicates brake close and 1 indicates brake open
    
#     carseg_group = seq[seq['carseg_flag'] == 1].groupby((seq['carseg_flag'] != 1).cumsum())
    
#     for k, v in carseg_group:
        
#         Relvl_list.append(v)  # Relevling cycle list

#     # 删除Releveling segment中小于1s的cycle        
#     Relvl_list = list(filter(lambda x: len(x)>20, Relvl_list))            

#     if len(Relvl_list)>0 and (num_Door == 0 or num_Door == np.nan): # start failure
        
#         log_text = {
#             "time": str(end_time),
#             "status ID": 2.9,
#             "event": "Start failure",                    
#             "description": "Lift cannot start successfully "
#                 }
#         event_list.append(log_text)      


            
#     DoorStat_text = {
#         'start_time':start_time,        
#         'end_time':end_time,
#         'duration':duration,        
#         'hour':hour,
#         'DoorI_peak':round(DoorI_peak,2),
#         'num_Door':num_Door,
#         'DoorOpen_Duration':round(DoorOpen_Duration,2),
#         'Stop_F': Stop_F
#         } 
    
#     DoorStat_text = {k: str(DoorStat_text[k]) if pd.isnull(DoorStat_text[k]) else DoorStat_text[k] for k in DoorStat_text }

#     return DoorStat_text, event_list

# def do_action_QEBL8(dataset, paras, CarSeg_list, DoorSeg_list):
#     event_list = []
#     CarStat_list = []
#     DoorStat_list = []
       
#     AI_FLAG = paras['AI_FLAG']
#     # Floor = paras['Floor']

#     ##计算loss
#     # criterion = torch.nn.MSELoss(reduction='sum')
#     # criterion2 = torch.nn.L1Loss(reduction='sum')
#     ############################################################
#     ################  1. CarSeg_list ########################
#     ############################################################
#     if len(CarSeg_list) > 0:
        
#         # brake_list = []
#         # brake_result = []
#         # motor_list = []
#         # motor_result = []
#         for i in range(len(CarSeg_list)):
#             seq = CarSeg_list[i]
#             # end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            
#             ####################### Rules #########################
#             CarStat_text,event_list = Stats_PerCarSeg_QEBL8(seq,paras,event_list)


#             CarStat_list.append(CarStat_text) 


#     ############################################################
#     ################  2. DoorSeg_list ########################
#     ############################################################
#     if len(DoorSeg_list) > 0:

#         # door_list = []
#         # door_result = []
#         for i in range(len(DoorSeg_list)):
#             seq = DoorSeg_list[i]        
#             # end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
#              ####################### Rules #########################
            
#             DoorStat_text,event_list = Stats_PerDoorSeg_QEBL8(seq,paras,event_list)
    
#             DoorStat_list.append(DoorStat_text) 
            

#     return CarStat_list, DoorStat_list, event_list



            
def DailyParas_Calculate_QEBL8(dataset,paras): # dataset为1s sample!!! 计算日参数，包括runtime,idletime,door open time,lock time,inspection time, rmu offline time, safety trip time, voltage dip time.
    print('************')
    print(dataset['Time'][0])
    #Date = dataset['Time'][0].strftime('%Y-%m-%d')
    Date = datetime.strptime(dataset['Time'][0], '%Y-%m-%d %H:%M:%S.%f').date()
    dataset = dataset.astype({'Brake':'float','Safety':'float','Resv-3':'float'})
    
    duration_offline = dataset['Brake'].isnull().sum()/60/60
    
        
    seq_lock = dataset.loc[(dataset['Safety']>=paras['Locked_SafetyI_range'][0]) & (dataset['Safety']<=paras['Locked_SafetyI_range'][1])]
    duration_lock = len(seq_lock)/60/60 # in hours
    
    seq_trip = dataset.loc[(dataset['Safety']>=paras['Tripped_SafetyI_range'][0]) & (dataset['Safety']<=paras['Tripped_SafetyI_range'][1])]
    duration_trip = len(seq_trip)/60/60
    
    seq_dip = dataset.loc[(dataset['Safety']>=paras['Voltage_Dip_range'][0]) & (dataset['Safety']<=paras['Voltage_Dip_range'][1])]
    duration_dip = len(seq_dip)/60/60
    
    seq_carrun = dataset.loc[(dataset['Safety']>=paras['Run_SafetyI_range'][0]) & (dataset['Safety']<=paras['Run_SafetyI_range'][1])]
    duration_run = len(seq_carrun)/60/60
    avg_safetyI_run = np.mean(seq_carrun['Safety'])

    seq_caridle = dataset.loc[(dataset['Safety']>=paras['Idle_SafetyI_range'][0]) & (dataset['Safety']<=paras['Idle_SafetyI_range'][1])]
    duration_idle = len(seq_caridle)/60/60
    avg_safetyI_idle = np.mean(seq_caridle['Safety'])
    
    if pd.isnull(paras['thres_resv3']):
        seq_dooropen = dataset.loc[(dataset['Safety']>=paras['Dooropen_SafetyI_range'][0]) & (dataset['Safety']<=paras['Dooropen_SafetyI_range'][1])]
        duration_dooropen = len(seq_dooropen)/60/60
    else:
        seq_dooropen = dataset.loc[dataset['Resv-3']<=paras['thres_resv3']]
        duration_dooropen = len(seq_dooropen)/60/60
    avg_safetyI_dooropen = np.mean(seq_dooropen['Safety'])

    seq_inspectionRun = dataset.loc[(dataset['Safety']>=paras['InspectionRun_range'][0]) & (dataset['Safety']<=paras['InspectionRun_range'][1])]
    duration_inspection = len(seq_inspectionRun)/60/60
    
    Daily_paras = {'Data':Date,
                   'duration_offline':round(duration_offline,2),
                   'duration_lock':round(duration_lock,2),
                   'duration_trip':round(duration_trip,2),
                   'duration_dip':round(duration_dip,2),
                   'duration_run':round(duration_run,2),
                   'duration_idle':round(duration_idle,2),
                   'duration_dooropen':round(duration_dooropen,2),
                   'duration_inspection':round(duration_inspection,2),                       
                   'avg_safetyI_run':round(avg_safetyI_run,2),                       
                   'avg_safetyI_idle':round(avg_safetyI_idle,2),                       
                   'avg_safetyI_dooropen':round(avg_safetyI_dooropen,2)                       
                   }
                
        
    Daily_paras = {k: str(Daily_paras[k]) if pd.isnull(Daily_paras[k]) else Daily_paras[k] for k in Daily_paras }

    return Daily_paras

