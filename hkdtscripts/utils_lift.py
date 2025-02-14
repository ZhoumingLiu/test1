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
from torch import nn
import numpy as np
# from matplotlib import pyplot as plt
from tsmoothie.smoother import *
from sklearn.preprocessing import MinMaxScaler
from math import ceil, floor
# from keras.models import load_model
# from utils.layer_utils import AttentionLSTM
# from keras.models import model_from_yaml
from datetime import datetime
# from scipy.signal import find_peaks

# import tensorflow as tf
# from keras import backend as K
# K.clear_session()

import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#%%
class LSTM_AutoEncoder_Brake(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(LSTM_AutoEncoder_Brake, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.Encoder_LSTM = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)
        self.Decoder_LSTM = nn.LSTM(self.hidden_size, self.input_size, 2, batch_first=True)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        encoder_lstm, (n, c) = self.Encoder_LSTM(input_x,
                                                 ((torch.zeros(2, self.batch_size, self.hidden_size)),
                                                  (torch.zeros(2, self.batch_size, self.hidden_size))))
        decoder_lstm, (n, c) = self.Decoder_LSTM(encoder_lstm,
                                                 ((torch.zeros(2, self.batch_size, self.input_size)),
                                                  (torch.zeros(2, self.batch_size, self.input_size))))
        return decoder_lstm.squeeze()

class LSTM_AutoEncoder_Motor(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(LSTM_AutoEncoder_Motor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.Encoder_LSTM = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.Decoder_LSTM = nn.LSTM(self.hidden_size, self.input_size, batch_first=True)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        encoder_lstm, (n, c) = self.Encoder_LSTM(input_x,
                                                 ((torch.zeros(1, self.batch_size, self.hidden_size)),
                                                  (torch.zeros(1, self.batch_size, self.hidden_size))))
        decoder_lstm, (n, c) = self.Decoder_LSTM(encoder_lstm,
                                                 ((torch.zeros(1, self.batch_size, self.input_size)),
                                                  (torch.zeros(1, self.batch_size, self.input_size))))
        return decoder_lstm.squeeze()
    
class LSTM_AutoEncoder_Door(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_fc1, hidden_size_fc2, batch_size):
        super(LSTM_AutoEncoder_Door, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.hidden_size_fc1 = hidden_size_fc1
        self.hidden_size_fc2 = hidden_size_fc2
        self.Encoder_LSTM = nn.LSTM(self.input_size, self.hidden_size, num_layers=3, batch_first=True, dropout=0.4)
        self.Encoder_fc1 = nn.Linear(self.hidden_size, self.hidden_size_fc1)
        self.Encoder_fc2 = nn.Linear(self.hidden_size_fc1, self.hidden_size_fc2)
        self.Decoder_LSTM = nn.LSTM(self.hidden_size, self.input_size, num_layers=2, batch_first=True, dropout=0.2)
        self.Decoder_fc1 = nn.Linear(self.hidden_size_fc2, self.hidden_size_fc1)
        self.Decoder_fc2 = nn.Linear(self.hidden_size_fc1, self.hidden_size)
        self.relu = nn.ReLU()
        self.Decoder_out = nn.Linear(self.input_size, self.input_size)

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        encoder1_lstm_out, (n, c) = self.Encoder_LSTM(input_x,
                                                      ((torch.zeros(3, self.batch_size, self.hidden_size)),
                                                       (torch.zeros(3, self.batch_size, self.hidden_size))))
        encoder1_fc1_out = self.Encoder_fc1(encoder1_lstm_out)
        encoder1_fc2_in = self.relu(encoder1_fc1_out)
        encoder1_out = self.Encoder_fc2(encoder1_fc2_in)

        decoder1_fc1 = self.relu(self.Decoder_fc1(encoder1_out))
        decoder1_fc2 = self.relu(self.Decoder_fc2(decoder1_fc1))
        decoder_lstm, (n, c) = self.Decoder_LSTM(decoder1_fc2,
                                                 ((torch.zeros(2, self.batch_size, self.input_size)),
                                                  (torch.zeros(2, self.batch_size, self.input_size))))
        decoder_out = self.Decoder_out(decoder_lstm)
        decoder_out = self.relu(decoder_out)
        encoder2_lstm_out, (n, c) = self.Encoder_LSTM(decoder_out,
                                                      ((torch.zeros(3, self.batch_size, self.hidden_size)),
                                                       (torch.zeros(3, self.batch_size, self.hidden_size))))
        encoder2_fc1_out = self.Encoder_fc1(encoder2_lstm_out)
        encoder2_fc2_in = self.relu(encoder2_fc1_out)
        encoder2_out = self.Encoder_fc2(encoder2_fc2_in)
        return decoder_out.squeeze(), encoder1_out.squeeze(), encoder2_out.squeeze()


    
## 序列切割函数，返回 CarSeg_list, DoorSeg_list
def Data_segment(dataset, paras, thres_BrakeKF, ver): 
# thres_BrakeKF为判断是否舍弃首个和最后一个segment的条件。ver是版本号，
#ver = 1: 用brake信号切割door cycle;
#ver = 2: 用resv3信号切割door cycle; 
#ver = 3: DC motor 并且用resv3信号切割door cycle;
#ver = 4: DC motor 并且用resv3信号切割door cycle;


    dataset = dataset.reset_index()
    dataset = dataset.rename(columns={'index': 'Time'})
    # use Kalman Filtering to smooth the data 
    smoother = KalmanSmoother(component='level_trend',
                          component_noise={'level': 0.1, 'trend': 0.1})
    

    dataset.loc[dataset['Distance'] >= 100,'Distance'] = np.nan # remove the outliers
    dataset.loc[dataset['Distance'] == 0,'Distance'] = np.nan # remove the outliers

    # dataset['Velocity'] = np.abs(dataset['Distance'].diff() / 0.05)  # calculate the abs velocity
    # dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  

    dataset['Velocity'] = np.abs(dataset['Distance'].diff(periods=20))  # calculate the abs velocity
    
    # 如果brake通道的缺失率在10%以下，才做切分，否则返回空。
    if dataset['Brake'].isnull().sum()/dataset.shape[0] < 0.1:
        
        dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  
        
        if ver != 3:
            dataset.loc[dataset['Motor'] < 0.15,'Motor'] = 0
            dataset.loc[dataset['Resv-1'] < 0.1,'Resv-1'] = 0

        
        dataset.loc[dataset['Brake'] < 0.05,'Brake'] = 0
        dataset.loc[dataset['Door'] < 0.025,'Door'] = 0
        dataset.loc[dataset['Safety'] < 0,'Safety'] = 0
        dataset.loc[dataset['Resv-2'] < 0.1,'Resv-2'] = 0
        dataset.loc[dataset['Resv-3'] < 0,'Resv-3'] = 0
        dataset.loc[dataset['Distance'] < 0,'Distance'] = 0
        dataset.loc[dataset['Velocity'] < 0,'Velocity'] = 0
        
        values = dataset[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance','Velocity']]
        smooth_seq = smoother.smooth(values.T).smooth_data.T # all smoothed sequences, if nan for a whole column, it will be 0.   
        dataset[['Motor_KF','Brake_KF','Safety_KF','Door_KF','Resv1_KF','Resv2_KF','Resv3_KF','Dist_KF','Vel_KF']] = smooth_seq
        
        ###### Update on 2022-12-31 00:25 为了提升切割效率 ############
        ###### Update on 2023-01-05 01:25 为了看有无false alarm brake fault 出现 ############

        dataset['Motor_KF'] = dataset['Motor_KF'].apply(lambda x: np.nan if dataset['Motor'].isnull().all() else x)
        dataset['Brake_KF'] = dataset['Brake_KF'].apply(lambda x: np.nan if dataset['Brake'].isnull().all() else x)
        dataset['Safety_KF'] = dataset['Safety_KF'].apply(lambda x: np.nan if dataset['Safety'].isnull().all() else x)
        dataset['Door_KF'] = dataset['Door_KF'].apply(lambda x: np.nan if dataset['Door'].isnull().all() else x)
        dataset['Resv1_KF'] = dataset['Resv1_KF'].apply(lambda x: np.nan if dataset['Resv-1'].isnull().all() else x)        
        dataset['Resv2_KF'] = dataset['Resv2_KF'].apply(lambda x: np.nan if dataset['Resv-2'].isnull().all() else x)
        dataset['Resv3_KF'] = dataset['Resv3_KF'].apply(lambda x: np.nan if dataset['Resv-3'].isnull().all() else x)
        dataset['Dist_KF'] = dataset['Dist_KF'].apply(lambda x: np.nan if dataset['Distance'].isnull().all() else x)
        dataset['Vel_KF'] = dataset['Vel_KF'].apply(lambda x: np.nan if dataset['Velocity'].isnull().all() else x)
        
        ########################################################################

        if ver != 3:
            dataset.loc[dataset['Motor_KF'] < 0,'Motor_KF'] = 0
            dataset.loc[dataset['Resv1_KF'] < 0,'Resv1_KF'] = 0
            
        dataset.loc[dataset['Brake_KF'] < 0.15,'Brake_KF'] = 0
        dataset.loc[dataset['Safety_KF'] < 0,'Safety_KF'] = 0
        dataset.loc[dataset['Door_KF'] < 0,'Door_KF'] = 0
        dataset.loc[dataset['Resv2_KF'] < 0,'Resv2_KF'] = 0
        dataset.loc[dataset['Resv3_KF'] < 0,'Resv3_KF'] = 0
        dataset.loc[dataset['Dist_KF'] < 0,'Dist_KF'] = 0
        dataset.loc[dataset['Vel_KF'] < 0.01,'Vel_KF'] = 0
    
        CarSeg_list = []
        DoorSeg_list = []
    
        dataset['carseg_flag'] = np.sign(dataset['Brake_KF']-0.01) # -1 indicates brake close and 1 indicates brake open

        if ver==1:
            dataset['doorseg_flag'] = np.sign(0.01 - dataset['Brake_KF']) # -1 indicates brake open and 1 indicates brake close

        else:
            dataset['doorseg_flag'] = np.sign(paras['thres_resv3']-dataset['Resv-3']) # series with -1 and 1, -1 indicates door close and 1 indicates door open


        carseg_group = dataset[dataset['carseg_flag'] == 1].groupby((dataset['carseg_flag'] != 1).cumsum())
        doorseg_group = dataset[dataset['doorseg_flag'] == 1].groupby((dataset['doorseg_flag'] != 1).cumsum())
        
        for k, v in carseg_group:
            
            CarSeg_list.append(v)  # Car motion cycle list
    
        for k, v in doorseg_group:
            
            DoorSeg_list.append(v)  # Door motion cycle list
        
            
            
       # 删除第一个不完整的car cycle
        if len(CarSeg_list)>0:
            if CarSeg_list[0].iloc[0]['Brake_KF'] > thres_BrakeKF:
        
                del (CarSeg_list[0])
                
        # 删除最后一个不完整的car cycle
                
        if len(CarSeg_list)>0:            
            if CarSeg_list[-1].iloc[-1]['Brake_KF'] > thres_BrakeKF:
        
                del (CarSeg_list[-1])
    
                
        # 删除car segment中小于3s的cycle        
        CarSeg_list = list(filter(lambda x: len(x)>60, CarSeg_list))
        # 删除door segment中为空的cycle        
        DoorSeg_list = list(filter(lambda x: len(x)>0, DoorSeg_list))
        
        if ver!=1:
            # 删除door segment中小于5s的cycle        
            DoorSeg_list = list(filter(lambda x: len(x)>100, DoorSeg_list))        
        
        # 删除door segment中首尾segment     
        if len(DoorSeg_list)>0:
         
            if (DoorSeg_list[0].iloc[0]['Time'] == dataset['Time'].iloc[0]) & (DoorSeg_list[0].iloc[-1]['Time'] != dataset['Time'].iloc[-1]):
                del (DoorSeg_list[0])
                     
        if len(DoorSeg_list)>0:
      
            if (DoorSeg_list[-1].iloc[-1]['Time'] == dataset['Time'].iloc[-1]) & (len(DoorSeg_list[-1])<400): # 末尾小于20s的删除
                del (DoorSeg_list[-1])            
    
    
    else:
        CarSeg_list = []
        DoorSeg_list = []
    
    return CarSeg_list, DoorSeg_list

# 定义楼层转换函数
def convert_floor(floor):
    if floor == -1.0:
        return 'G/F'
    elif floor == None:
        return 'NAN/F'
    else:
        return f'{int(floor)}/F'
    
    
# 确定停止的楼层（停止状态）
def StopF_Caculation(paras,seq): 
    
    # if np.isnan(paras['Floor_height']).all() or np.isnan(seq['Distance']).all():    
    if paras['Position_sensor'] == 0: # for No sensor 
        
        Stop_F='NAN/F'

    elif paras['Position_sensor'] == 1: # for LiDAR   
    
        Stop_F = paras['Floor_level'][np.nanargmin(np.abs(seq['Distance'].iloc[-1] - np.array(paras['Floor_height'])))] # STOP floor of this cycle
    
    else: # for MMU  
    
        last_valid_index = seq['floor'].last_valid_index()
        
        last_valid_value = seq.loc[last_valid_index, 'floor'] if last_valid_index is not None else None
        
        Stop_F = convert_floor(last_valid_value)
        
    return Stop_F

# 计算电梯运行时交通参数
def Traffic_Caculation(paras,seq): 
    Traffic_dict = {
        'Speed_peak': None,
        'mileage': None,
        'Depart_F': None,
        'Arrive_F': None,
        'Fs_travelled': None,
        'F_Travelled': None,
        'Dir_Travelled': None,
        'position_dev': None
    }
        
    if paras['Position_sensor'] == 0: # for No sensor 
    
        Traffic_dict['Depart_F'] = Traffic_dict['Arrive_F'] = 'NAN/F'
        Traffic_dict['Speed_peak']=Traffic_dict['mileage']=Traffic_dict['Fs_travelled']=Traffic_dict['F_Travelled']=Traffic_dict['Dir_Travelled']=Traffic_dict['position_dev'] = np.nan   

    elif paras['Position_sensor'] == 1: # for LiDAR
        
        Depart_index = np.nanargmin(np.abs(seq['Distance'].iloc[0] - np.array(paras['Floor_height'])))
        Arrive_index = np.nanargmin(np.abs(seq['Distance'].iloc[-1] - np.array(paras['Floor_height'])))
        Traffic_dict['Depart_F'] = paras['Floor_level'][Depart_index]  # start floor of this cycle
        Traffic_dict['Arrive_F'] = paras['Floor_level'][Arrive_index]  # end floor of this cycle   
        Traffic_dict['position_dev'] = np.nanmin(np.abs(seq['Distance'].iloc[-10:].median() - np.array(paras['Floor_height']))) # stop position deviation from the desired floor
        Traffic_dict['Fs_travelled'] = Arrive_index - Depart_index # the floors the elevator travels, + upward, - downward
        Traffic_dict['F_Travelled'] = np.abs(Traffic_dict['Fs_travelled']) # the floors the elevator travels
        Traffic_dict['Dir_Travelled'] = np.sign(Traffic_dict['Fs_travelled']) # +1 upward, -1 downward, 0 releveling 
        Traffic_dict['Speed_peak'] = seq['Vel_KF'].max() # peak speed when car runs
        Traffic_dict['mileage'] = np.abs(seq['Distance'].iloc[-10:].median() - seq['Distance'].iloc[:10].median())
    
    else: # for MMU  
        Traffic_dict['Speed_peak'] = seq['velocity'].max()  # 这是MMU测量的velocity     
        Traffic_dict['position_dev'] = np.nan
    
        first_valid_index = seq['floor'].first_valid_index()       
        last_valid_index = seq['floor'].last_valid_index()
        
        if first_valid_index is not None and last_valid_index is not None: 
            
            first_valid_floor = seq.loc[first_valid_index, 'floor'] 
            Traffic_dict['Depart_F'] = convert_floor(first_valid_floor)
            
            if first_valid_floor == -1:
                first_valid_floor = 0

            last_valid_floor = seq.loc[last_valid_index, 'floor']      
            Traffic_dict['Arrive_F'] = convert_floor(last_valid_floor)
           
            if last_valid_floor == -1:
                last_valid_floor = 0
                
            Traffic_dict['Fs_travelled'] = last_valid_floor - first_valid_floor # the floors the elevator travels, + upward, - downward
            Traffic_dict['F_Travelled'] = np.abs(Traffic_dict['Fs_travelled']) # the floors the elevator travels
            Traffic_dict['Dir_Travelled'] = np.sign(Traffic_dict['Fs_travelled']) # +1 upward, -1 downward, 0 releveling 

        else:
            Traffic_dict['Depart_F'] = Traffic_dict['Arrive_F'] = 'NAN/F'
            Traffic_dict['Fs_travelled']=Traffic_dict['F_Travelled']=Traffic_dict['Dir_Travelled'] = np.nan
            
        # 找到第一个非 nan 值的索引
        first_valid_index = seq['mileage'].first_valid_index()
        
        # 找到最后一个非 nan 值的索引
        last_valid_index = seq['mileage'].last_valid_index()
        
        # 使用这些索引获取值，并计算里程差
        if first_valid_index is not None and last_valid_index is not None:
            Traffic_dict['mileage'] = seq.loc[last_valid_index, 'mileage'] - seq.loc[first_valid_index, 'mileage']
        else:
            Traffic_dict['mileage'] = np.nan

    return Traffic_dict

# 计算电流相关特征
def Current_Calculation(paras,seq): 
    Current_dict = {
        'MotorI_peak': None,
        'MotorI_start': None,
        'MotorI_brake': None,
        'MotorI_steady': None,
        'Resv1I_peak': None,
        'Resv1I_start': None,
        'Resv1I_brake': None,
        'Resv1I_steady': None,
        'Resv2I_peak': None,
        'Resv2I_start': None,
        'Resv2I_brake': None,
        'Resv2I_steady': None,
        'BrakeI_min': None,
        'BrakeI_peak': None,
        'BrakeI_steady': None,
        'SafetyI_run': None,
        'SafetyI_max': None,
        'Resv3I_run': None
    }
    
    
    Current_dict['BrakeI_steady'] = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current
    Current_dict['BrakeI_peak'] = seq['Brake_KF'].max() # peak brake current when car runs 
    
    Current_dict['SafetyI_run'] = seq['Safety'].mean() #average safety current when car runs
    Current_dict['SafetyI_max'] = seq['Safety'].max() #max safety current when car runs
        
    Current_dict['Resv3I_run'] = seq['Resv-3'].mean() # average resv3 current when car runs    
    ###################################################
    ################  电机为AC的情况   ##################
    ###################################################
    if paras['Motor_type'] == 'AC':
        
        Current_dict['MotorI_peak'] = seq['Motor_KF'].max()  # peak current of motor
        Current_dict['Resv1I_peak'] = seq['Resv1_KF'].max()  # peak current of resv1
        Current_dict['Resv2I_peak'] = seq['Resv2_KF'].max()  # peak current of resv2
    
        if len(seq) >= 200: # 行程大于10s
            Current_dict['MotorI_start'] = seq['Motor_KF'].iloc[25:75].max()  # starting current of motor
            Current_dict['MotorI_brake'] = seq['Motor_KF'].iloc[-75:-25].max()  # braking current of motor
            Current_dict['MotorI_steady'] = seq['Motor_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # steady current of motor
    
            Current_dict['Resv1I_start'] = seq['Resv1_KF'].iloc[25:75].max()  # starting current of resv1
            Current_dict['Resv1I_brake'] = seq['Resv1_KF'].iloc[-75:-25].max()  # braking current of resv1
            Current_dict['Resv1I_steady'] = seq['Resv1_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # steady current of Resv1
    
            Current_dict['Resv2I_start'] = seq['Resv2_KF'].iloc[25:75].max()  # starting current of resv2
            Current_dict['Resv2I_brake'] = seq['Resv2_KF'].iloc[-75:-25].max()  # braking current of resv2
            Current_dict['Resv2I_steady'] = seq['Resv2_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # steady current of Resv2
    
            # BrakeI_steady = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current   
            # BrakeI_peak = seq['Brake_KF'].max() # brake peak current
            Current_dict['BrakeI_min'] = seq['Brake'].iloc[30:-30].min() # brake minimum current 
        
        
        elif 80 < len(seq) < 200: # 行程在 4s-10s之间
            Current_dict['MotorI_start'] = seq['Motor_KF'].iloc[25:75].max()  # starting current of motor
            Current_dict['MotorI_brake'] = seq['Motor_KF'].iloc[-75:-25].max()  # braking current of motor
            Current_dict['MotorI_steady'] = np.nan  # steady current of motor        
            
            Current_dict['Resv1I_start'] = seq['Resv1_KF'].iloc[25:75].max()  # starting current of resv1
            Current_dict['Resv1I_brake'] = seq['Resv1_KF'].iloc[-75:-25].max()  # braking current of resv1
            Current_dict['Resv1I_steady'] = np.nan   # steady current of Resv1
    
            Current_dict['Resv2I_start'] = seq['Resv2_KF'].iloc[25:75].max()  # starting current of resv2
            Current_dict['Resv2I_brake'] = seq['Resv2_KF'].iloc[-75:-25].max()  # braking current of resv2
            Current_dict['Resv2I_steady'] = np.nan   # steady current of Resv2
            
            # BrakeI_steady = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current
            # BrakeI_peak = seq['Brake_KF'].max() # brake peak current   
            Current_dict['BrakeI_min'] = seq['Brake'].iloc[30:-30].min() # brake minimum current 
                
        else:
            Current_dict['MotorI_start'] = np.nan  # starting current of motor
            Current_dict['MotorI_brake'] = np.nan  # braking current of motor
            Current_dict['MotorI_steady'] = np.nan  # steady current of motor
    
            Current_dict['Resv1I_start'] = np.nan  # starting current of resv1
            Current_dict['Resv1I_brake'] = np.nan  # braking current of resv1
            Current_dict['Resv1I_steady'] = np.nan   # steady current of Resv1
    
            Current_dict['Resv2I_start'] = np.nan  # starting current of resv2
            Current_dict['Resv2I_brake'] = np.nan  # braking current of resv2
            Current_dict['Resv2I_steady'] = np.nan   # steady current of Resv2        
    
            # BrakeI_steady = np.nan  # brake steady current
            # BrakeI_peak = np.nan  # brake peak current
            Current_dict['BrakeI_min'] = seq['Brake'].iloc[5:-5].min() # brake minimum current 
            

    ###################################################
    ################  电机为DC的情况   ##################
    ###################################################
        
    elif paras['Motor_type'] == 'DC':
        
        Current_dict['Resv1I_peak'] = seq['Resv1_KF'].abs().max()  # Armature peak current
        Current_dict['MotorI_brake'] = np.nan
        Current_dict['MotorI_steady'] = np.nan
        Current_dict['Resv2I_start'] = np.nan
        Current_dict['Resv2I_peak'] = np.nan
        Current_dict['Resv2I_brake'] = np.nan
        Current_dict['Resv2I_steady'] = np.nan
        
        if len(seq) >= 240: # 行程大于12s
    
            Current_dict['MotorI_start'] = seq['Motor_KF'].iloc[:10].mean()  # Full Field current
            Current_dict['MotorI_peak'] = seq['Motor_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # Run Field current
    
            Current_dict['Resv1I_start'] = seq['Resv1_KF'].iloc[40:70].mean()  # Armature starting current
            Current_dict['Resv1I_brake'] = seq['Resv1_KF'].iloc[-50:-25].mean()  # Armature braking current
            Current_dict['Resv1I_steady'] = seq['Resv1_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # Armature steady current
            
            Current_dict['BrakeI_min'] = seq['Brake'].iloc[30:-30].min() # minimum brake current when car runs 
    
    
        elif 80 < len(seq) < 240: # 行程在 4s-12s之间
    
            Current_dict['MotorI_start'] = seq['Motor_KF'].iloc[:10].mean()  # Full Field current
            Current_dict['MotorI_peak'] = np.nan  # Run Field current
    
            
            Current_dict['Resv1I_start'] = seq['Resv1_KF'].iloc[40:70].mean()  # Armature starting current
            Current_dict['Resv1I_brake'] = seq['Resv1_KF'].iloc[-50:-25].mean()  # Armature braking current
            Current_dict['Resv1I_steady'] = np.nan  # Armature steady current
            
            Current_dict['BrakeI_min'] = seq['Brake'].iloc[30:-30].min() # minimum brake current when car runs 
                    
        else:
            
            Current_dict['MotorI_start'] = np.nan  # Full Field current
            Current_dict['MotorI_peak'] = np.nan  # Run Field current
    
            
            Current_dict['Resv1I_start'] = np.nan  # Armature starting current
            Current_dict['Resv1I_brake'] = np.nan  # Armature braking current
            Current_dict['Resv1I_steady'] = np.nan  # Armature steady current
            
            Current_dict['BrakeI_min'] = seq['Brake'].iloc[5:-5].min() # minimum brake current when car runs 
            
    return Current_dict

# 门的开关次数
def num_DoorOpenClose(paras,series,seq):
    if paras['Position_sensor'] < 3:
        
        if not pd.isnull(paras['line_Door']):
            
            group = [] 
            mask = series > paras['line_Door']
            group_keys = mask[mask == True].groupby((mask != True).cumsum())
            for k, v in group_keys:
                group.append(v)  # Car motion cycle list
            num_Door = len(group)           

            
            if paras['DoorWaveform_type'] == 1: # 一个波峰 = 一对
                # peaks, _ = find_peaks(series, height = paras['line_Door'])
                # num_Door = ceil(len(peaks))  # pairs of door open&closes
                num_Door = num_Door

            elif paras['DoorWaveform_type'] == 2: # 两个波峰 = 一对
                # peaks, _ = find_peaks(series, height = paras['line_Door'],distance = 50)
                # num_Door = ceil(len(peaks)/2)  # pairs of door open&closes
                num_Door = floor(num_Door/2)
    
                
            elif paras['DoorWaveform_type'] == 4: # 四个波峰 = 一对
                # peaks, _ = find_peaks(series, height = paras['line_Door'])
                # num_Door = ceil(len(peaks)/4)  # pairs of door open&closes
                num_Door = floor(num_Door/4)
                
                
            # elif paras['DoorWaveform_type'] == 0: # 是矩形pattern
    
            #     num_Door = len(np.argwhere(np.diff(np.sign(paras['line_Door']-series))))/4 # number of door closing and opening pairs
            #     num_Door = floor(num_Door)
    
        else:
            num_Door = np.nan  
            
    else: # MMU+S6003
    
    # 找到第一个非 nan 值的索引
        first_valid_index = seq['openCloseDoorNum'].first_valid_index()
        
        # 找到最后一个非 nan 值的索引
        last_valid_index = seq['openCloseDoorNum'].last_valid_index()
        
        # 使用这些索引获取值，并计算里程差
        if first_valid_index is not None and last_valid_index is not None:
            num_Door = seq.loc[last_valid_index, 'openCloseDoorNum'] - seq.loc[first_valid_index, 'openCloseDoorNum']
        else:
            num_Door = np.nan
    
    return num_Door

# 计算门相关的重要参数
def Door_Calculation(paras,seq): 

    if paras['Position_sensor'] < 3: # 没装S6003的电梯,都用电流信号去判断
    
        # 0-用整个doorseg
        # 1-用safety信号再去切割出开关门信号
        # 2-用resv3信号再去切割出开关门信号
        
        if paras['DoorOpen_ref'] == 1: #用safety信号去判断
        
            mask = (seq['Safety'] > paras['Dooropen_SafetyI_range'][0]) & (seq['Safety'] < paras['Dooropen_SafetyI_range'][1])

            seq_dooropen = seq.loc[mask, 'Door_KF']
            
            num_Door = num_DoorOpenClose(paras,seq_dooropen,seq)
            
            DoorOpen_Duration = len(seq_dooropen)/20
        
        elif paras['DoorOpen_ref'] == 2: #用resv3信号去判断

            mask = seq['Resv-3'] < paras['thres_resv3']
                
            seq_dooropen = seq.loc[mask, 'Door_KF']
            
            num_Door = num_DoorOpenClose(paras,seq_dooropen,seq)
            
            DoorOpen_Duration = len(seq_dooropen)/20            
                
        else: #用整个doorseg
        
            seq_dooropen = seq['Door_KF']
            
            num_Door = num_DoorOpenClose(paras,seq_dooropen,seq)
            
            DoorOpen_Duration = len(seq_dooropen)/20                 
            
            
    else: #如果装了S6003,直接用S6003返回的开关门累计次数去计算
        
        # 找到第一个非 nan 值的索引
        first_valid_index = seq['openCloseDoorNum'].first_valid_index()
        
        # 找到最后一个非 nan 值的索引
        last_valid_index = seq['openCloseDoorNum'].last_valid_index()
        
        # 使用这些索引获取值，并计算里程差
        if first_valid_index is not None and last_valid_index is not None:
            num_Door = seq.loc[last_valid_index, 'openCloseDoorNum'] - seq.loc[first_valid_index, 'openCloseDoorNum']
        else:
            num_Door = np.nan        

        # value_counts = seq['door'].value_counts()
        
        # # 获取数值为 3 和 4 的数量
        # count_3 = value_counts.get(3, 0)  # 如果没有 3，则返回 0
        # count_4 = value_counts.get(4, 0)  # 如果没有 4，则返回 0
        
        # # 计算总数
        # DoorOpen_Duration = (count_3 + count_4)/20 # update on 2024-2-17        
        filtered_seq = seq[(seq['door'] > 0) & (seq['door'] < 5)]
        
        # 计算这些行的数量
        DoorOpen_Duration = len(filtered_seq)/20
        
    return num_Door, DoorOpen_Duration

# 检测有无releveling事件发生
def Releveling(paras, seq):
    
    Relvl_list = []

    seq['carseg_flag'] = np.sign(seq['Brake_KF']-0.01) # -1 indicates brake close and 1 indicates brake open
    
    carseg_group = seq[seq['carseg_flag'] == 1].groupby((seq['carseg_flag'] != 1).cumsum())
    
    for k, v in carseg_group:
        
        Relvl_list.append(v)  # Relevling cycle list

    # 删除Releveling segment中小于1s的cycle        
    Relvl_list = list(filter(lambda x: len(x)>20, Relvl_list))       
    
    
    return Relvl_list

# 1 - RMU offline
def RMU_offline(paras, dataset,event_list,end_time):
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
        
    return event_list

# 2.1 - UCM
def UCM(paras, seq, mileage, DoorI_peak, Stop_F, event_list, end_time):
    
    if mileage > 100.3 and DoorI_peak > paras['line_Door']:
        log_text = {
            "time": str(end_time),
            "status ID": 2.1,
            "event": "UCM",
            "description": "lift car has travelled beyond door zone (+/- 300 mm of floor level)",
            "floor": Stop_F,
            "delsign":0  
                }
        event_list.append(log_text)
        
    return event_list

# 2.2 - ACO  
def ACO(paras, seq, Traffic_dict, event_list, end_time):
    
    if Traffic_dict['Dir_Travelled'] == 1:  # 判断电梯是否处于上升状态
        if paras['Position_sensor'] >= 2: # for MMU     
            len_overspeed = 20*len(seq.loc[seq['velocity'] > 1.15 * paras['RatedSpeed']]) # When lift car is travelling upward and >115% of rated speed
        
        else:
            len_overspeed = len(seq.loc[seq['Velocity'] > 1.15 * paras['RatedSpeed']]) 

        if len_overspeed > 80: # overspeed 超过4s
            log_text = {
                "time": str(end_time),
                "status ID": 2.2,
                "event": "ACO",
                "description": "Lift ascending over speed ",
                "floor": Traffic_dict['Arrive_F'],
                "delsign":0                   
                 }
            event_list.append(log_text)   
    return event_list

# 2.3 - Brake faults
def Brake_Faults(paras, seq, Current_dict, Traffic_dict, event_list, end_time):    

    BrakeI_steady = Current_dict['BrakeI_steady']
    BrakeI_peak = Current_dict['BrakeI_peak']
    

    if not paras['BrIsteady_Range'][0] < BrakeI_steady < paras['BrIsteady_Range'][1] and ~np.isnan(BrakeI_steady): # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
             "description": "anomaly brake steady current magnitude:" + str(round(BrakeI_steady,2)) + " A",            
             "floor": Traffic_dict['Arrive_F'],
             "delsign":0
                 }
         event_list.append(log_text)

    if not paras['BrIpeak_Range'][0] < BrakeI_peak < paras['BrIpeak_Range'][1] and seq['Brake'].isnull().sum()==0: # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
             "description": "anomaly brake peak current magnitude:" + str(round(BrakeI_peak,2)) + " A",
             "floor": Traffic_dict['Arrive_F'],
             "delsign":0
                 }
         event_list.append(log_text) 
         
    return event_list


# 2.6 - Safety Tripped (Idle)
def SafetyTrip(dataset,paras): 
    event_list = []
    
    # scenario 1 
    if paras['SafetyTrip_FLAG'] == 1:
        
        seq_trip = dataset.loc[(dataset['Safety']>=paras['Tripped_SafetyI_range'][0]) & (dataset['Safety']<=paras['Tripped_SafetyI_range'][1])]
        if len(seq_trip)>=120:  
    
            Stop_F = StopF_Caculation(paras,seq_trip)    
                
            log_text = {
                "time": str(seq_trip.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
                "status ID": 2.6,
                "event": "Safety tripped (Idle)",
                "description": "Urgent - Safety tripped when lift is in idle",
                "floor": Stop_F,
                "delsign":0  
                    }
            event_list.append(log_text) # 把trip事件存到event_list里
            
    # scenario 2     
    elif paras['SafetyTrip_FLAG'] == 2:
        
        seq_trip1 = dataset.loc[(dataset['Safety']>=paras['Tripped_SafetyI_range'][0]) & (dataset['Safety']<=paras['Tripped_SafetyI_range'][1])]
        seq_trip2 = dataset.loc[(dataset['Door']>=paras['Tripped_SafetyI_range'][2]) & (dataset['Door']<=paras['Tripped_SafetyI_range'][3])]
    
        if (len(seq_trip1)>=120) & (len(seq_trip2)>=120):
            
    
            Stop_F = StopF_Caculation(paras,seq_trip1)    
                
            log_text = {
                "time": str(seq_trip1.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
                "status ID": 2.6,
                "event": "Safety tripped (Idle)",
                "description": "Urgent - Safety tripped when lift is in idle",
                "floor": Stop_F,
                "delsign":0  
                    }
            event_list.append(log_text) # 把trip事件存到event_list里  

    # scenario 3                 
    elif paras['SafetyTrip_FLAG'] == 3:

        TripSeg_list = []
    
        dataset['trip_flag'] = np.sign(dataset['Safety']-paras['Tripped_SafetyI_range'][1]) # -1 indicates safety trip and 1 indicates safety not trip
        
        tripseg_group = dataset[dataset['trip_flag'] == -1].groupby((dataset['trip_flag'] != -1).cumsum())
        
        for k, v in tripseg_group:
            
            TripSeg_list.append(v)  # Trip list
        
        TripSeg_list = list(filter(lambda x: len(x)>60, TripSeg_list)) #保留trip 时间大于3s的seg
               
        if len(TripSeg_list)>0: 
            
            Stop_F = StopF_Caculation(paras,dataset)    
                 
            log_text = {
                "time": str(TripSeg_list[-1].index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
                "status ID": 2.6,
                "event": "Safety tripped (Idle)",
                "description": "Urgent - Safety tripped when lift is in idle",
                "floor": Stop_F,
                "delsign":0  
                    }
            event_list.append(log_text) # 把trip事件存到event_list里          
    
    return event_list




# 2.7 - Stop at Non-service Floor (原来叫作 Over Travel)     
def Stop_NonserviceFloor(paras, seq, Traffic_dict, event_list, end_time):
    if 0.3 < Traffic_dict['position_dev'] < 7 and seq['Velocity'].iloc[-1] < 0.5 and seq['Motor_KF'].iloc[-1] < 10:
        log_text = {
            "time": str(end_time),
            "status ID": 2.7,
            "event": "Stop at Non-service Floor",                    
            "description": "Lift stops at the level not at the proper position (>+/- 300mm)",
            "floor": Traffic_dict['Arrive_F'],
            "delsign":0            
                }
        event_list.append(log_text)      

    return event_list

# 2.8 - Sudden Stop     
#用该trip的最后0.5s速度的中位数或者距离偏移 判断以及最后一个Motor Current值判断
def Sudden_Stop(paras, seq, Traffic_dict, event_list, end_time):   
    if seq['Motor_KF'].iloc[-1] > 10 and (0.5 < seq['Velocity'].iloc[-10:].median() < 3 or 0.3 < Traffic_dict['position_dev'] < 7):        
        log_text = {
            "time": str(end_time),
            "status ID": 2.8,
            "event": "Sudden stop",                    
            "description": "Lift suddenly stops at the mid of travelling ",
            "floor": Traffic_dict['Arrive_F'],
            "delsign":0            
                }
        event_list.append(log_text) 
    
    return event_list

# 2.9 - Start failure     
def Start_Failure(paras, Current_dict, Traffic_dict, event_list, end_time):
    if Traffic_dict['mileage'] < 0.2 and Traffic_dict['Arrive_F'] != 'NAN/F' and Current_dict['Resv3I_run'] > paras['thres_resv3']: # (1)没移动；（2）Arrive_F 有值; (3) Resv3 是处于电梯运行的level
        log_text = {
            "time": str(end_time),
            "status ID": 2.9,
            "event": "Start failure",                    
            "description": "Lift cannot start successfully ",
            "floor": Traffic_dict['Arrive_F'],
            "delsign":0            
                }
        event_list.append(log_text)  
        
    return event_list

def Start_Failure_Door(paras, seq, num_Door, Stop_F, Relvl_list, event_list, end_time):

    if len(Relvl_list)>0 and seq['Brake_KF'].iloc[0] < 0.2 and (num_Door == 0 or num_Door == np.nan): # start failure，update on 2023-1-11, add one more condition for special case such as QM and QE lift
        
        log_text = {
            "time": str(end_time),
            "status ID": 2.9,
            "event": "Start failure",                    
            "description": "Lift cannot start successfully ",
            "floor": Stop_F,
            "delsign":0              
                }
        event_list.append(log_text)          
      
    return event_list

# 3.1 - Door anomaly
def Door_Anomaly(paras, seq, num_Door, DoorOpen_Duration, DoorI_peak, Stop_F, event_list, end_time):
    
    if DoorOpen_Duration > paras['thres_DoorOpenDuration'] and Stop_F != 'G/F': 

        if paras['RMU_ID'] != 14: # disable the long door openning  duration alert for QML3 (update on 2023-02-20)
            
            log_text = {
                "time": str(end_time),
                "status ID": 3.1,
                "event": "Door anomaly",
                "description": "Door openning time lasts longer than " + str(round(DoorOpen_Duration,2)) + " s at " + Stop_F,
                "floor": Stop_F,
                "delsign":0  
                    }
            event_list.append(log_text)   
        
    if DoorI_peak < paras['DrIpeak_Range'][0] or DoorI_peak > paras['DrIpeak_Range'][1]: #2023-2-22 从非AI model里提取出来
    # if not paras['DrIpeak_Range'][0] < DoorI_peak < paras['DrIpeak_Range'][1] and ~np.isnan(DoorI_peak): # out of normal current range of door
        
        log_text = {
            "time": str(end_time),
            "status ID": 3.1,
            "event": "Door anomaly",
            "description": "Anomaly door motor current magnitude:" + str(round(DoorI_peak,2)) + " A",
            "floor": Stop_F,
            "delsign":0                  
                }
        event_list.append(log_text) 
            
    # if paras['AI_FLAG'] == 0: # when AI is not adopted

    if num_Door >= paras['thres_numDoor']:
        
        
        log_text = {
            "time": str(end_time),
            "status ID": 3.1,
            "event": "Door anomaly",
            "description": "Excessive door open & close actions",
            "floor": Stop_F,
            "delsign":0                  
                }
        event_list.append(log_text)    
    
    ######  Update on 2023-2-10  #####      
    if paras['RMU_ID']==14: # 对于QML3 需要通过判断电梯停止时 motor电流fluctuate的时间/频率来判断门是存在故障
        # seq['Motor'][seq['Motor'] > 1].count() 
        seq['MotorWave_flag'] = np.sign(seq['Motor']-1) 
        MotorWave_count = len(seq[seq['MotorWave_flag'] == 1].groupby((seq['MotorWave_flag'] != 1).cumsum())) # motor 电流fluctuate的次数   
        if MotorWave_count > 70:
            
            log_text = {
                "time": str(end_time),
                "status ID": 3.1,
                "event": "Door anomaly",
                "description": "Irregular door pattern identified at " + Stop_F, # update 23-9-20
                "floor": Stop_F,
                "delsign":0  
                    }
            event_list.append(log_text)      
    
    return event_list

# 3.2 - Brake unsmooth operation
def Brake_UnsmoothOpertion(paras, seq, Current_dict, Traffic_dict, event_list, end_time):
#  Brake ramp down, safety surge up and motor rise a little 
    if (Current_dict['BrakeI_min'] < paras['BrIsteady_Range'][0]) & (Current_dict['SafetyI_max'] > 2.5*paras["Run_SafetyI_range"][1]):
        log_text = {
            "time": str(end_time),
            "status ID": 3.2,
            "event": "Brake unsmooth operation",
            "description": "Brake unsmooth operation",
            "floor": Traffic_dict['Arrive_F'],
            "delsign":0
                }
        event_list.append(log_text)      
    
    return event_list


# 3.3 - Motor anomaly
def Motor_Anomaly(paras, Current_dict, Traffic_dict, event_list, end_time):   
    
    MotorI_peak = Current_dict['MotorI_peak']
    MotorI_start = Current_dict['MotorI_start']
    Resv1I_peak = Current_dict['Resv1I_peak']
    Resv1I_start = Current_dict['Resv1I_start']
    Resv1I_brake = Current_dict['Resv1I_brake']
    Resv1I_steady = Current_dict['Resv1I_steady']
    
    if paras['Motor_type'] == 'AC':
        
        if not np.isnan(MotorI_peak):

            if not paras['MotIpeak_Range'][0] < MotorI_peak < paras['MotIpeak_Range'][1]: # out of the normal motor peak current range
                 log_text = {
                     "time": str(end_time),
                     "status ID": 3.3,
                     "event": "Motor anomaly",
                     "description": "anomaly motor peak current magnitude:" + str(round(MotorI_peak,2)) + " A",
                     "floor": Traffic_dict['Arrive_F'],
                     "delsign":0
                         }
                 event_list.append(log_text) 
             
    elif paras['Motor_type'] == 'DC':
         
        # Motor field current anomaly
        if not np.isnan(MotorI_peak):
            
            if not paras['RunField_Range'][0] < MotorI_peak < paras['RunField_Range'][1]: 
                 log_text = {
                     "time": str(end_time),
                     "status ID": 3.3,
                     "event": "Motor anomaly",
                     "description": "anomaly run field current magnitude:" + str(round(MotorI_peak,2)) + " A",
                     "floor": Traffic_dict['Arrive_F'],
                     "delsign":0             
                         }
                 event_list.append(log_text) 
                 
        if not np.isnan(MotorI_start):
            if not paras['FullField_Range'][0] < MotorI_start < paras['FullField_Range'][1]: # out of Full field current
                 log_text = {
                     "time": str(end_time),
                     "status ID": 3.3,
                     "event": "Motor anomaly",
                     "description": "anomaly full field current magnitude:" + str(round(MotorI_start,2)) + " A",
                     "floor": Traffic_dict['Arrive_F'],
                     "delsign":0      
                         }
                 event_list.append(log_text) 
             
             
        # Motor armature current anomaly
        if not np.isnan(Resv1I_peak):

            if not paras['ArmaturePeak_Range'][0] < Resv1I_peak < paras['ArmaturePeak_Range'][1]: # out of Armature Peak current
                 log_text = {
                     "time": str(end_time),
                     "status ID": 3.3,
                     "event": "Motor anomaly",
                     "description": "anomaly armature peak current magnitude:" + str(round(Resv1I_peak,2)) + " A",
                     "floor": Traffic_dict['Arrive_F'],
                     "delsign":0  
                         }
                 event_list.append(log_text) 
      
        if not np.isnan(Resv1I_start):
    
            if not paras['ArmatureStart_Range'][0] < Resv1I_start < paras['ArmatureStart_Range'][1]: # out of Armature Starting current
                 log_text = {
                     "time": str(end_time),
                     "status ID": 3.3,
                     "event": "Motor anomaly",
                     "description": "anomaly armature starting current magnitude:" + str(round(Resv1I_start,2)) + " A",
                     "floor": Traffic_dict['Arrive_F'],
                     "delsign":0  
                         }
                 event_list.append(log_text) 
                 
                 
        if not np.isnan(Resv1I_brake):
        
            if not paras['ArmatureBrake_Range'][0] < Resv1I_brake < paras['ArmatureBrake_Range'][1]: # out of Armature Braking current
                 log_text = {
                     "time": str(end_time),
                     "status ID": 3.3,
                     "event": "Motor anomaly",
                     "description": "anomaly armature braking current magnitude:" + str(round(Resv1I_brake,2)) + " A",
                     "floor": Traffic_dict['Arrive_F'],
                     "delsign":0  
                         }
                 event_list.append(log_text) 
     
        if not np.isnan(Resv1I_steady):
    
            if not paras['ArmatureSteady_Range'][0] < Resv1I_steady < paras['ArmatureSteady_Range'][1]: # out of Armature Steady current
                 log_text = {
                     "time": str(end_time),
                     "status ID": 3.3,
                     "event": "Motor anomaly",
                     "description": "anomaly armature steady current magnitude:" + str(round(Resv1I_steady,2)) + " A",
                     "floor": Traffic_dict['Arrive_F'],
                     "delsign":0  
                         }
                 event_list.append(log_text)
                
    return event_list


# 4 - Inspection     
def Inspection(paras, seq, Traffic_dict, event_list, end_time):
    if paras['Position_sensor'] >= 2: # for MMU    
        if 2 in seq['workMode'].values:
            log_text = {
                "time": str(end_time),
                "status ID": 4,
                "event": "Inspection mode",                    
                "description": "Inspection mode - maintenance (by MMU)",
                "floor": Traffic_dict['Arrive_F'],
                "delsign":0            
                    }            
    else: # for LiDAR
        if 0.1 < Traffic_dict['Speed_peak'] < 0.5 and Traffic_dict['Arrive_F'] != 'NAN/F':  # 判断是否为慢车检修模式    
            log_text = {
                "time": str(end_time),
                "status ID": 4,
                "event": "Inspection mode",                    
                "description": "Inspection mode - maintenance",
                "floor": Traffic_dict['Arrive_F'],
                "delsign":0            
                    }
            event_list.append(log_text) 
            
    return event_list


# 5 - Lock mode
def LockMode(dataset,paras):
    event_list = []
    seq_lock = dataset.loc[(dataset['Safety']>=paras['Locked_SafetyI_range'][0]) & (dataset['Safety']<=paras['Locked_SafetyI_range'][1])]    

    if len(seq_lock)>=1000:
        # if np.isnan(paras['Floor_height']).all() or np.isnan(seq_lock['Distance']).all():        
        #     Stop_F=np.nan
        # else:
        #     Stop_F = paras['Floor_level'][np.nanargmin(np.abs(seq_lock['Distance'].iloc[-1] - np.array(paras['Floor_height'])))] # STOP floor of this cycle
        Stop_F = StopF_Caculation(paras,seq_lock)    
        
        log_text = {
            "time": str(seq_lock.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 5,
            "event": "Locked",
            "description": "Lock mode - out of service",
            "floor": Stop_F,
            "delsign":0              
                            }
        event_list.append(log_text) # 把锁机事件存到event_list里  
        
    return event_list



def Stats_PerCarSeg(seq, paras, event_list): # Statistic information for each car segment/cycle

    ############################################################
    ##################  1. Basic Information ###################
    ############################################################
    start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
    hour = seq.iloc[-1]['Time'].hour                 
    
    
    ############################################################
    ################  2. Key Parameters Calculation ############
    ############################################################
    Traffic_dict = Traffic_Caculation(paras,seq)  

    Current_dict = Current_Calculation(paras,seq)
    

    ############################################################
    ############  3. Event Detection based on Rules ############
    ############################################################
        
    # 2.2-ACO  
    event_list = ACO(paras, seq, Traffic_dict, event_list, end_time)
    
    # 2.3-Brake Faults        
    event_list = Brake_Faults(paras, seq, Current_dict, Traffic_dict, event_list, end_time)
    
    # 2.7-Stop at Non-service Floor (原来叫作 Over Travel)     
    # event_list = Stop_NonserviceFloor(paras, seq, Traffic_dict, event_list, end_time)

    # 2.8-Sudden Stop (2024-1-11 disable)    
    # event_list = Sudden_Stop(paras, seq, Traffic_dict, event_list, end_time) 

    # 2.9-Start failure (2024-1-11 disable)    
    # event_list = Start_Failure(paras, Current_dict, Traffic_dict, event_list, end_time)  
    
    # 3.2-Brake Unsmooth Operation          
    event_list = Brake_UnsmoothOpertion(paras, seq, Current_dict, Traffic_dict, event_list, end_time)
    
    # 3.3-Motor anomaly       
    event_list = Motor_Anomaly(paras, Current_dict, Traffic_dict, event_list, end_time) 
         
    # 4-Inspection     
    event_list = Inspection(paras, seq, Traffic_dict, event_list, end_time)


         
    CarStat_text = {
        'start_time':start_time,
        'end_time':end_time,
        'duration':duration,
        'hour':hour,
        'mileage':round(Traffic_dict['mileage'],2),
        'Depart_F':Traffic_dict['Depart_F'],
        'Arrive_F':Traffic_dict['Arrive_F'],
        'Fs_travelled':Traffic_dict['Fs_travelled'],
        'F_Travelled':Traffic_dict['F_Travelled'],
        'Dir_Travelled':Traffic_dict['Dir_Travelled'],
        'MotorI_start':round(Current_dict['MotorI_start'],2), # DC: Full Field current
        'MotorI_peak':round(Current_dict['MotorI_peak'],2),   # DC: Run Field current
        'MotorI_brake':round(Current_dict['MotorI_brake'],2), 
        'MotorI_steady':round(Current_dict['MotorI_steady'],2),         
        'BrakeI_steady':round(Current_dict['BrakeI_steady'],2),
        'BrakeI_peak':round(Current_dict['BrakeI_peak'],2),
        'SafetyI_run':round(Current_dict['SafetyI_run'],2),
        'Resv1I_start':round(Current_dict['Resv1I_start'],2), # DC:Armature Starting current
        'Resv1I_peak':round(Current_dict['Resv1I_peak'],2), # DC:Armature Peak current
        'Resv1I_brake':round(Current_dict['Resv1I_brake'],2),  # DC:Armature Brake current
        'Resv1I_steady':round(Current_dict['Resv1I_steady'],2), # DC:Armature Steady current          
        'Resv2I_start':round(Current_dict['Resv2I_start'],2),
        'Resv2I_peak':round(Current_dict['Resv2I_peak'],2),
        'Resv2I_brake':round(Current_dict['Resv2I_brake'],2), 
        'Resv2I_steady':round(Current_dict['Resv2I_steady'],2),        
        'Resv3I_run':round(Current_dict['Resv3I_run'],2),
        'Speed_peak':round(Traffic_dict['Speed_peak'],2)
        } 
    
    CarStat_text = {k: str(CarStat_text[k]) if pd.isnull(CarStat_text[k]) else CarStat_text[k] for k in CarStat_text }
            
    return CarStat_text, event_list



def Stats_PerDoorSeg(seq, paras, event_list): # Statistic information for each car segment/cycle
    ############################################################
    ################  1. Basic Information ###################
    ############################################################
    start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
    hour = seq.iloc[-1]['Time'].hour                    

    ############################################################
    ################  2. Key Parameters Calculation ############
    ############################################################   
    # distance = np.abs(seq.iloc[-1]['Distance']-seq.iloc[0]['Distance'])

    # num_Door = num_DoorOpenClose(paras,seq)  
    num_Door, DoorOpen_Duration = Door_Calculation(paras, seq)
    
    if num_Door > 0:
        DoorI_peak = seq['Door_KF'].max()
    else:
        DoorI_peak = np.nan
        
              
    if paras['Position_sensor'] == 0: # for No sensor 
    
        mileage = np.nan   
        
    elif paras['Position_sensor'] == 1: # for LiDAR

        mileage = np.abs(seq['Distance'].iloc[-10:].median() - seq['Distance'].iloc[:10].median()) 
    
    else: # for MMU  
        # 找到第一个非 nan 值的索引
        first_valid_index = seq['mileage'].first_valid_index()
        
        # 找到最后一个非 nan 值的索引
        last_valid_index = seq['mileage'].last_valid_index()
        
        # 使用这些索引获取值，并计算里程差
        if first_valid_index is not None and last_valid_index is not None:
            mileage = seq.loc[last_valid_index, 'mileage'] - seq.loc[first_valid_index, 'mileage']
        else:
            mileage = np.nan
            
    
    # DoorOpen_Duration = DoorTime_Calculation(paras,seq)
        
   
    Stop_F = StopF_Caculation(paras,seq)


    ############################################################
    ############  3. Event Detection based on Rules ############
    ###################################s########################
    
    # UCM   
    event_list = UCM(paras, seq, mileage, DoorI_peak, Stop_F, event_list, end_time)
   
    # door anomaly   
    event_list = Door_Anomaly(paras, seq, num_Door, DoorOpen_Duration, DoorI_peak, Stop_F, event_list, end_time) 
     

    # Start Failure   (2024-1-11 disable)
    
    # Relvl_list = Releveling(paras, seq)  #### 检测是否有releveling case 发生 ####           

    # event_list = Start_Failure_Door(paras, seq, num_Door, Stop_F, Relvl_list, event_list, end_time)        


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



def LSTMFCN_Motor(seq,end_time,model_name, event_list):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            K.set_session(sess)
            
            if model_name == 'EMSD_V1':
                model = model_from_yaml(open('/app/hkdtspts/devpy/inspection_model.yaml').read(), custom_objects={"AttentionLSTM": AttentionLSTM})
                model.load_weights('/app/hkdtspts/devpy/inspection_model_weights.h5', by_name=False)
                # model = model_from_yaml(open('./devpy/inspection_model.yaml').read(), custom_objects={"AttentionLSTM": AttentionLSTM})
                # model.load_weights('./devpy/inspection_model_weights.h5', by_name=False)
            else:
                model = load_model('/app/hkdtspts/devpy/'+ model_name,custom_objects={"AttentionLSTM": AttentionLSTM}, compile=False)                        
                # model = load_model('./devpy/'+ model_name,custom_objects={"AttentionLSTM": AttentionLSTM}, compile=False)        
                
            X_mat = np.array(seq[['Motor']])
            
            ##### Update on 2022-12-31 20:10:00 ####
            if len(X_mat)>1200:
                X_mat = X_mat[0:1200]
            ########################################
            
            
            X_test = np.zeros((1, X_mat.shape[1], 1200))  # num_samples x num_variables x num_timesteps
            X_test[0, :, :X_mat.shape[0]] = np.transpose(X_mat)
            pred = model.predict(X_test, batch_size = 128)                                    
        
    pred_label = np.argmax(pred, axis=1)

    if pred_label != 0:
        
        log_text = {
            "time": str(end_time),
            "status ID": 4,
            "event": "Inspection mode",                    
            "description": "Inspection mode - maintenance",
            "floor": 'nan',
            "delsign":0  
                }
        event_list.append(log_text)  
    return event_list

def LSTMFCN_Door(seq,end_time,Stop_F, model_name, event_list): # 2023-2-22 add stop_F and model_name
         
    X_mat = np.array(seq[['Door_KF']]) 
    ##### Update on 2022-12-30 22:44:00 ####
    if len(X_mat)>1200:
        X_mat = X_mat[0:1200]
    ########################################
    
    X_test = np.zeros((1, X_mat.shape[1], 1200))  # num_samples x num_variables x num_timesteps
    X_test[0, :, :X_mat.shape[0]] = np.transpose(X_mat)
    X_test = torch.tensor(X_test).float()

    with torch.no_grad():
    
        model = torch.load('./devpy/' + model_name, map_location='cpu')
        
        model.eval()
        pred = model(X_test)                                    
    
        
    pred_label = np.argmax(pred.detach().numpy(), axis=1)
    if pred_label != 0:

        log_text = {
            "time": str(end_time),
            "status ID": 3.1,
            "event": "Door anomaly",                    
            "description": "Door anomaly open & close",
            "floor": Stop_F,
            "delsign":0  
                }
        event_list.append(log_text)  
    return event_list




## 执行数据分析和AI主程序

def do_action(dataset, paras, CarSeg_list, DoorSeg_list):
    event_list = []
    CarStat_list = []
    DoorStat_list = []
       

    ############################################################
    ################  1. CarSeg_list ########################
    ############################################################
    if len(CarSeg_list) > 0:
        

        for i in range(len(CarSeg_list)):
            seq = CarSeg_list[i]
            end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            
            ####################### Car Rules #########################

            
            CarStat_text,event_list = Stats_PerCarSeg(seq,paras,event_list)
            
            CarStat_list.append(CarStat_text) 

            ####################### Motor AI Model #########################

            if paras['AI_Motor_Model'] != '': # Trigger the  Motor AI module 
            

                seq.loc[seq['Motor'] < 0, 'Motor'] = 0
                model_name = paras['AI_Motor_Model']
                event_list = LSTMFCN_Motor(seq,end_time,model_name,event_list)

                ########### ########### ########### ########### ########### ###########
    else:
        end_time = dataset.index[-1].strftime('%Y-%m-%d %H:%M:%S.%f')

                 

    ############################################################
    ################  2. DoorSeg_list ########################
    ############################################################
    if len(DoorSeg_list) > 0:


        for i in range(len(DoorSeg_list)):
            seq = DoorSeg_list[i]        
            end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
             ####################### Door Rules #########################
            
            DoorStat_text,event_list = Stats_PerDoorSeg(seq,paras,event_list)
    
            DoorStat_list.append(DoorStat_text) 

            ####################### Door AI Model #########################
            
            if paras['AI_Door_Model'] != '': # Trigger the  Door AI module 
            
                Stop_F = DoorStat_text['Stop_F']
                model_name = paras['AI_Door_Model']
                event_list = LSTMFCN_Door(seq,end_time,Stop_F,model_name,event_list)
                
            

    return CarStat_list, DoorStat_list, event_list


            
def DailyParas_Calculate(dataset,paras): # dataset为1s sample!!! 计算日参数，包括runtime,idletime,door open time,lock time,inspection time, rmu offline time, safety trip time, voltage dip time.
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

def final_output(dataset,paras,last_status,event_list,CarStat_list,DoorStat_list):
    ##### 1 - 状态及事件表 ###### 
    result = {
        "Lift_ID": paras['RMU_ID'],
        "Lift_Name": paras['Lift_Name'],
        "last_status": last_status,
        "event_list": event_list,
        "period": {
            "start": dataset.index[0],
            "end": dataset.index[-1]
                },
        "post_time":pd.to_datetime(datetime.now())
            }
    
     ##### 2 - 电梯运行关键指标统计表 ###### 
    CarSeg_Stats = {
        "Lift_ID": paras['RMU_ID'],
        "Lift_Name": paras['Lift_Name'],
        "Nos_Run": len(CarStat_list),
        "CarStat_list": CarStat_list,
        "period": {
            "start": dataset.index[0],
            "end": dataset.index[-1]
                },
            }
     ##### 3 - 电梯门动作关键指标统计表 ###### 
    DoorSeg_Stats = {
        "Lift_ID": paras['RMU_ID'],
        "Lift_Name": paras['Lift_Name'],
        "Nos_DoorSeg": len(DoorStat_list),
        "DoorStat_list": DoorStat_list,
        "period": {
            "start": dataset.index[0],
            "end": dataset.index[-1]
                },
            }    
    return result, CarSeg_Stats, DoorSeg_Stats

