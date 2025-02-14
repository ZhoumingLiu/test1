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
from matplotlib import pyplot as plt
from tsmoothie.smoother import *
from sklearn.preprocessing import MinMaxScaler
from math import ceil
from keras.models import load_model
from utils.layer_utils import AttentionLSTM
from keras.models import model_from_yaml
from datetime import datetime
from scipy.signal import find_peaks

import tensorflow as tf
from keras import backend as K
K.clear_session()

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
#“1”代表用brake信号切割door cycle; “2”代表用resv3信号切割door cycle; "3"代表DC motor 并且用resv3信号切割door cycle

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

        # Update on 2022-12-19， 为了用safety+resv3 判断QML3的门运行
        # if ver==1:
        #     dataset['doorseg_flag'] = np.sign(0.01 - dataset['Brake_KF']) # -1 indicates brake open and 1 indicates brake close

        # elif ver==2: # QM CASE
        #     dataset['doorseg_flag'] = np.sign(paras['thres_resv3']-dataset['Resv-3']) # series with -1 and 1, -1 indicates door close and 1 indicates door open
        #     dataset['doorseg_flag'] = np.sign(paras['Dooropen_SafetyI_range'][1]-dataset['Safety']) # series with -1 and 1, -1 indicates door close and 1 indicates door open
        #     #xxxxxx- 这段还没完
        # elif ver==3: # QE CASE
        #     dataset['doorseg_flag'] = np.sign(paras['thres_resv3']-dataset['Resv-3']) # series with -1 and 1, -1 indicates door close and 1 indicates door open            


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





## 数据预处理,针对LSTM-AE模块
def pre_data(dataframe, input_size=1200, batch_size=128):
    l1 = [0]
    l1 = pd.DataFrame(l1)
    while dataframe.shape[1] < input_size:
        dataframe[dataframe.shape[1] + 1] = l1
    if dataframe.shape[1] > input_size:
        dataframe = dataframe.iloc[:, :input_size]
    count = dataframe.shape[0] % batch_size
    if count != 0:
        count = 128 - count
        dataframe = np.concatenate([dataframe, np.zeros(shape=(count, 1200))])
    dataframe = np.array(dataframe)
    min_max = MinMaxScaler(feature_range=(0, 1))
    dataframe = min_max.fit_transform(dataframe.T)
    dataframe = dataframe.T
    dataframe = pd.DataFrame(dataframe)
    dataframe.fillna(0, inplace=True)
    dataframe = np.array(dataframe)
    dataframe = torch.FloatTensor(dataframe)
    return dataframe

def Stats_PerCarSeg_AC(seq, paras, event_list): # Statistic information for each car segment/cycle (AC type)

    ############################################################
    ################  1. Traffic Information ###################
    ############################################################
    start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
    hour = seq.iloc[-1]['Time'].hour                 
    # mileage = np.abs(seq['Distance'].iloc[-1] - seq['Distance'].iloc[0])

    mileage = np.abs(seq['Distance'].iloc[-10:].median() - seq['Distance'].iloc[:10].median())
    
    if np.isnan(paras['Floor']).all():        
        Depart_F = Arrive_F = Fs_travelled = F_Travelled = Dir_Travelled = position_dev = np.nan
    else:
        Depart_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor'])) - 1 # start floor of this cycle
        Arrive_F = np.nanargmin(np.abs(seq['Distance'].iloc[-1] - paras['Floor'])) - 1 # end floor of this cycle        
        # position_dev = np.nanmin(np.abs(seq['Distance'].iloc[-1] - paras['Floor'])) # stop position deviation from the desired floor
        position_dev = np.nanmin(np.abs(seq['Distance'].iloc[-10:].median() - paras['Floor'])) # stop position deviation from the desired floor
        Fs_travelled = Arrive_F - Depart_F # the floors the elevator travels, + upward, - downward
        F_Travelled = np.abs(Fs_travelled) # the floors the elevator travels
        Dir_Travelled = np.sign(Fs_travelled) # +1 upward, -1 downward, 0 releveling
    
    ############################################################
    ################  2. Key Parameters Calculation ############
    ############################################################
    MotorI_peak = seq['Motor_KF'].max()  # peak current of motor
    Resv1I_peak = seq['Resv1_KF'].max()  # peak current of resv1
    Resv2I_peak = seq['Resv2_KF'].max()  # peak current of resv2

    if len(seq) >= 200: # trip 大于10s
        MotorI_start = seq['Motor_KF'].iloc[25:75].max()  # starting current of motor
        MotorI_brake = seq['Motor_KF'].iloc[-75:-25].max()  # braking current of motor
        MotorI_steady = seq['Motor_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # steady current of motor

        Resv1I_start = seq['Resv1_KF'].iloc[25:75].max()  # starting current of resv1
        Resv1I_brake = seq['Resv1_KF'].iloc[-75:-25].max()  # braking current of resv1
        Resv1I_steady = seq['Resv1_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # steady current of Resv1

        Resv2I_start = seq['Resv2_KF'].iloc[25:75].max()  # starting current of resv2
        Resv2I_brake = seq['Resv2_KF'].iloc[-75:-25].max()  # braking current of resv2
        Resv2I_steady = seq['Resv2_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # steady current of Resv2

        # BrakeI_steady = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current   
        # BrakeI_peak = seq['Brake_KF'].max() # brake peak current
        BrakeI_min = seq['Brake'].iloc[30:-30].min() # brake minimum current 
    
    
    elif 80 < len(seq) < 200: # trip在 4s-10s之间
        MotorI_start = seq['Motor_KF'].iloc[25:75].max()  # starting current of motor
        MotorI_brake = seq['Motor_KF'].iloc[-75:-25].max()  # braking current of motor
        MotorI_steady = np.nan  # steady current of motor        
        
        Resv1I_start = seq['Resv1_KF'].iloc[25:75].max()  # starting current of resv1
        Resv1I_brake = seq['Resv1_KF'].iloc[-75:-25].max()  # braking current of resv1
        Resv1I_steady = np.nan   # steady current of Resv1

        Resv2I_start = seq['Resv2_KF'].iloc[25:75].max()  # starting current of resv2
        Resv2I_brake = seq['Resv2_KF'].iloc[-75:-25].max()  # braking current of resv2
        Resv2I_steady = np.nan   # steady current of Resv2
        
        # BrakeI_steady = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current
        # BrakeI_peak = seq['Brake_KF'].max() # brake peak current   
        BrakeI_min = seq['Brake'].iloc[30:-30].min() # brake minimum current 

        
    else:
        MotorI_start = np.nan  # starting current of motor
        MotorI_brake = np.nan  # braking current of motor
        MotorI_steady = np.nan  # steady current of motor

        Resv1I_start = np.nan  # starting current of resv1
        Resv1I_brake = np.nan  # braking current of resv1
        Resv1I_steady = np.nan   # steady current of Resv1

        Resv2I_start = np.nan  # starting current of resv2
        Resv2I_brake = np.nan  # braking current of resv2
        Resv2I_steady = np.nan   # steady current of Resv2        

        # BrakeI_steady = np.nan  # brake steady current
        # BrakeI_peak = np.nan  # brake peak current
        BrakeI_min = seq['Brake'].iloc[5:-5].min() # brake minimum current 
        
    BrakeI_steady = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current
    BrakeI_peak = seq['Brake_KF'].max() # peak brake current 
    
    SafetyI_run = seq['Safety'].mean() #average safety current when car runs
    SafetyI_max = seq['Safety'].max() #max safety current when car runs
    

    Resv3I_run = seq['Resv-3'].mean() # average resv3 current when car runs
        
    Speed_peak = seq['Vel_KF'].max() # peak speed  when car runs
    
    
    ############################################################
    ############  3. Event Detection based on Rules ############
    ############################################################
    # Motor anomaly       # out of the normal motor current range
    if not paras['MotIpeak_Range'][0] < MotorI_peak < paras['MotIpeak_Range'][1]: # out of the normal motor peak current range
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly motor peak current magnitude:" + str(round(MotorI_peak,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0
                 }
         event_list.append(log_text) 
         
    # Brake Faults         # out of the normal steady brake current range
    if not paras['BrIsteady_Range'][0] < BrakeI_steady < paras['BrIsteady_Range'][1] and ~np.isnan(BrakeI_steady): # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
             "description": "anomaly brake steady current magnitude:" + str(round(BrakeI_steady,2)) + " A",            
             "floor": str(Arrive_F),
             "delsign":0
                 }
         event_list.append(log_text)

    # Brake Faults          # out of the normal peak brake current range
    if not paras['BrIpeak_Range'][0] < BrakeI_peak < paras['BrIpeak_Range'][1] and seq['Brake'].isnull().sum()==0: # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
             "description": "anomaly brake peak current magnitude:" + str(round(BrakeI_peak,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0
                 }
         event_list.append(log_text)       

    
    # Brake Unsmooth Operation          #  Brake ramp down, safety surge up and motor rise a little 
    if (BrakeI_min < paras['BrIsteady_Range'][0]) & (SafetyI_max > 2.5*paras["Run_SafetyI_range"][1]):
        log_text = {
            "time": str(end_time),
            "status ID": 3.2,
            "event": "Brake unsmooth operation",
            "description": "Brake unsmooth operation",
            "floor": str(Arrive_F),
            "delsign":0
                }
        event_list.append(log_text) # 把brake异常存到event_list里        
        

    # ACO     
    if Dir_Travelled == 1:  # 判断电梯是否处于上升状态

        seq_overspeed = seq.loc[seq['Velocity'] > 1.15 * paras['RatedSpeed']] # When lift car is travelling upward and >115% of rated speed

        if len(seq_overspeed) > 80: # overspeed 超过4s
            log_text = {
                "time": str(end_time),
                "status ID": 2.2,
                "event": "ACO",
                "description": "Lift ascending over speed ",
                "floor": str(Arrive_F),
                "delsign":0                   
                 }
            event_list.append(log_text)    
            
    # Inspection     
    if 0.1 < Speed_peak < 0.5 and ~np.isnan(Arrive_F):  # 判断是否为慢车检修模式
        log_text = {
            "time": str(end_time),
            "status ID": 4,
            "event": "Inspection mode",                    
            "description": "Inspection mode - maintenance",
            "floor": str(Arrive_F),
            "delsign":0            
                }
        event_list.append(log_text)    

    # Stop at Non-service Floor (原来叫作 Over Travel)     
    if 0.3 < position_dev < 7 and seq['Velocity'].iloc[-1] < 0.5 and seq['Motor_KF'].iloc[-1] < 10:
        log_text = {
            "time": str(end_time),
            "status ID": 2.7,
            "event": "Stop at Non-service Floor",                    
            "description": "Lift stops at the level not at the proper position (>+/- 300mm)",
            "floor": str(Arrive_F),
            "delsign":0            
                }
        event_list.append(log_text)  

    # Sudden Stop     
    # if 0.5 < seq['Velocity'].iloc[-10:].median() < 3 and seq['Motor_KF'].iloc[-1] > 10:  #用该trip的最后1s速度的中位数判断以及最后一个Motor Current值判断    
    #用该trip的最后0.5s速度的中位数或者距离偏移 判断以及最后一个Motor Current值判断   
    if seq['Motor_KF'].iloc[-1] > 10 and (0.5 < seq['Velocity'].iloc[-10:].median() < 3 or 0.3 < position_dev < 7):        
        log_text = {
            "time": str(end_time),
            "status ID": 2.8,
            "event": "Sudden stop",                    
            "description": "Lift suddenly stops at the mid of travelling ",
            "floor": str(Arrive_F),
            "delsign":0            
                }
        event_list.append(log_text)  

    # Start failure     
    if mileage < 0.2 and ~np.isnan(Arrive_F) and Resv3I_run > paras['thres_resv3']: # (1)没移动；（2）Arrive_F 有值; (3) Resv3 是处于电梯运行的level
    # if seq['Distance'].diff().max()<0.01 and ~np.isnan(Arrive_F) and Resv3I_run > paras['thres_resv3']: # (1)没有任何位移变化(没法判断)；（2）Arrive_F 有值; (3) Resv3 是处于电梯运行的level

        log_text = {
            "time": str(end_time),
            "status ID": 2.9,
            "event": "Start failure",                    
            "description": "Lift cannot start successfully ",
            "floor": str(Arrive_F),
            "delsign":0            
                }
        event_list.append(log_text)  

         
    CarStat_text = {
        'start_time':start_time,
        'end_time':end_time,
        'duration':duration,
        'hour':hour,
        'mileage':round(mileage,2),
        'Depart_F':Depart_F,
        'Arrive_F':Arrive_F,
        'Fs_travelled':Fs_travelled,
        'F_Travelled':F_Travelled,
        'Dir_Travelled':Dir_Travelled,
        'MotorI_start':round(MotorI_start,2),
        'MotorI_peak':round(MotorI_peak,2),   
        'MotorI_brake':round(MotorI_brake,2), 
        'MotorI_steady':round(MotorI_steady,2),         
        'BrakeI_steady':round(BrakeI_steady,2),
        'BrakeI_peak':round(BrakeI_peak,2),
        'SafetyI_run':round(SafetyI_run,2),
        'Resv1I_start':round(Resv1I_start,2),
        'Resv1I_peak':round(Resv1I_peak,2),
        'Resv1I_brake':round(Resv1I_brake,2), 
        'Resv1I_steady':round(Resv1I_steady,2),          
        'Resv2I_start':round(Resv2I_start,2),
        'Resv2I_peak':round(Resv2I_peak,2),
        'Resv2I_brake':round(Resv2I_brake,2), 
        'Resv2I_steady':round(Resv2I_steady,2),        
        'Resv3I_run':round(Resv3I_run,2),
        'Speed_peak':round(Speed_peak,2)
        } 
    
    CarStat_text = {k: str(CarStat_text[k]) if pd.isnull(CarStat_text[k]) else CarStat_text[k] for k in CarStat_text }
            
    return CarStat_text, event_list

def Stats_PerCarSeg_DC(seq, paras, event_list): # Statistic information for each car segment/cycle

    ############################################################
    ################  1. Traffic Information ###################
    ############################################################
    start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
    hour = seq.iloc[-1]['Time'].hour                 
    # mileage = np.abs(seq['Distance'].iloc[-1] - seq['Distance'].iloc[0])

    mileage = np.abs(seq['Distance'].iloc[-10:].median() - seq['Distance'].iloc[:10].median())
    
    if np.isnan(paras['Floor']).all():        
        Depart_F = Arrive_F = Fs_travelled = F_Travelled = Dir_Travelled = position_dev = np.nan
    else:
        Depart_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor'])) - 1 # start floor of this cycle
        Arrive_F = np.nanargmin(np.abs(seq['Distance'].iloc[-1] - paras['Floor'])) - 1 # end floor of this cycle        
        # position_dev = np.nanmin(np.abs(seq['Distance'].iloc[-1] - paras['Floor'])) # stop position deviation from the desired floor
        position_dev = np.nanmin(np.abs(seq['Distance'].iloc[-10:].median() - paras['Floor'])) # stop position deviation from the desired floor
        Fs_travelled = Arrive_F - Depart_F # the floors the elevator travels, + upward, - downward
        F_Travelled = np.abs(Fs_travelled) # the floors the elevator travels
        Dir_Travelled = np.sign(Fs_travelled) # +1 upward, -1 downward, 0 releveling
    
    ############################################################
    ################  2. Key Parameters Calculation ############
    ############################################################
    
    Resv1I_peak = seq['Resv1_KF'].abs().max()  # Armature peak current


    if len(seq) >= 240: # trip 大于12s

        MotorI_start = seq['Motor_KF'].iloc[:10].mean()  # Full Field current
        MotorI_peak = seq['Motor_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # Run Field current

        Resv1I_start = seq['Resv1_KF'].iloc[40:70].mean()  # Armature starting current
        Resv1I_brake = seq['Resv1_KF'].iloc[-50:-25].mean()  # Armature braking current
        Resv1I_steady = seq['Resv1_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean()  # Armature steady current
        
        BrakeI_min = seq['Brake'].iloc[30:-30].min() # minimum brake current when car runs 


    elif 80 < len(seq) < 240: # trip在 4s-12s之间

        MotorI_start = seq['Motor_KF'].iloc[:10].mean()  # Full Field current
        MotorI_peak = np.nan  # Run Field current

        
        Resv1I_start = seq['Resv1_KF'].iloc[40:70].mean()  # Armature starting current
        Resv1I_brake = seq['Resv1_KF'].iloc[-50:-25].mean()  # Armature braking current
        Resv1I_steady = np.nan  # Armature steady current
        
        BrakeI_min = seq['Brake'].iloc[30:-30].min() # minimum brake current when car runs 


        
    else:
        
        MotorI_start = np.nan  # Full Field current
        MotorI_peak = np.nan  # Run Field current

        
        Resv1I_start = np.nan  # Armature starting current
        Resv1I_brake = np.nan  # Armature braking current
        Resv1I_steady = np.nan  # Armature steady current
        
        BrakeI_min = seq['Brake'].iloc[5:-5].min() # minimum brake current when car runs 
      
        
    BrakeI_steady = seq['Brake_KF'].iloc[(round(len(seq)/2)-15):(round(len(seq)/2)+15)].mean() # brake steady current
    BrakeI_peak = seq['Brake_KF'].max() # peak brake current when car runs 
    
    SafetyI_run = seq['Safety'].mean() #average safety current when car runs
    SafetyI_max = seq['Safety'].max() #max safety current when car runs
    

    Resv3I_run = seq['Resv-3'].mean() # average resv3 current when car runs
        
    Speed_peak = seq['Vel_KF'].max() # peak speed  when car runs   
    
    
    
        
    ############################################################
    ############  3. Event Detection based on Rules ############
    ############################################################
    # Motor field current anomaly      
    if not paras['RunField_Range'][0] < MotorI_peak < paras['RunField_Range'][1] and ~np.isnan(MotorI_peak): # out of Run field current
         log_text = {
             
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly run field current magnitude:" + str(round(MotorI_peak,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0             
                 }
         event_list.append(log_text) 

    if not paras['FullField_Range'][0] < MotorI_start < paras['FullField_Range'][1] and ~np.isnan(MotorI_start): # out of Full field current
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly full field current magnitude:" + str(round(MotorI_start,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0      
                 }
         event_list.append(log_text) 
         
         
    # Motor armature current anomaly 
    if not paras['ArmaturePeak_Range'][0] < Resv1I_peak < paras['ArmaturePeak_Range'][1] and ~np.isnan(Resv1I_peak): # out of Armature Peak current
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly armature peak current magnitude:" + str(round(Resv1I_peak,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0  
                 }
         event_list.append(log_text) 

    if not paras['ArmatureStart_Range'][0] < Resv1I_start < paras['ArmatureStart_Range'][1] and ~np.isnan(Resv1I_start): # out of Armature Starting current
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly armature starting current magnitude:" + str(round(Resv1I_start,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0  
                 }
         event_list.append(log_text) 

    if not paras['ArmatureBrake_Range'][0] < Resv1I_brake < paras['ArmatureBrake_Range'][1] and ~np.isnan(Resv1I_brake): # out of Armature Braking current
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly armature braking current magnitude:" + str(round(Resv1I_brake,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0  
                 }
         event_list.append(log_text) 

    if not paras['ArmatureSteady_Range'][0] < Resv1I_steady < paras['ArmatureSteady_Range'][1] and ~np.isnan(Resv1I_steady): # out of Armature Steady current
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly armature steady current magnitude:" + str(round(Resv1I_steady,2)) + " A",
             "floor": str(Arrive_F),
             "delsign":0  
                 }
         event_list.append(log_text)
         
    # Brake Faults         # out of the normal brake current range
    if not paras['BrIsteady_Range'][0] < BrakeI_steady < paras['BrIsteady_Range'][1] and ~np.isnan(BrakeI_steady): # out of the normal brake current range
        
        if duration > 5: # update on 2023-3-20, 5s以上的trip才判断，不然很可能是releveling 
             log_text = {
                 "time": str(end_time),
                 "status ID": 2.3,
                 "event": "Brake Faults",
                 "description": "anomaly brake steady current magnitude:" + str(round(BrakeI_steady,2)) + " A",
                 "floor": str(Arrive_F),
                 "delsign":0  
                     }
             event_list.append(log_text)

    # Brake Faults          # out of the normal brake current range
    if not paras['BrIpeak_Range'][0] < BrakeI_peak < paras['BrIpeak_Range'][1] and duration > 5: # out of the normal brake current range,update on 2023-3-20, 5s以上的trip才判断，不然很可能是releveling     
        
        # if duration > 5: # update on 2023-3-20, 5s以上的trip才判断，不然很可能是releveling     
            log_text = {
                "time": str(end_time),
                "status ID": 2.3,
                "event": "Brake Faults",
                "description": "anomaly brake peak current magnitude:" + str(round(BrakeI_peak,2)) + " A",
                "floor": str(Arrive_F),
                "delsign":0  
                    }
            event_list.append(log_text)       

        # Brake Unsmooth Operation          #  Brake ramp down, safety surge up and motor rise a little 
    if (BrakeI_min < paras['BrIsteady_Range'][0]) & (SafetyI_max > 2.5*paras["Run_SafetyI_range"][1]):
        log_text = {
            "time": str(end_time),
            "status ID": 3.2,
            "event": "Brake unsmooth operation",
            "description": "Brake unsmooth operation",
            "floor": str(Arrive_F),
            "delsign":0  
                }
        event_list.append(log_text) # 把brake异常存到event_list里        
        
    # ACO     
    if Dir_Travelled == 1:  # 判断电梯是否处于上升状态

        seq_overspeed = seq.loc[seq['Velocity'] > 1.15 * paras['RatedSpeed']] # When lift car is travelling upward and >115% of rated speed

        if len(seq_overspeed) > 80: # overspeed 超过4s
            log_text = {
                "time": str(end_time),
                "status ID": 2.2,
                "event": "ACO",
                "description": "Lift ascending over speed ",
                "floor": str(Arrive_F),
                "delsign":0                    
                }
            event_list.append(log_text)    
            
    # Inspection     
    if 0.1 < Speed_peak < 0.5 and ~np.isnan(Arrive_F):  # 判断是否为慢车检修模式
        log_text = {
            "time": str(end_time),
            "status ID": 4,
            "event": "Inspection mode",                    
            "description": "Inspection mode - maintenance",
            "floor": str(Arrive_F),
            "delsign":0                  
            }
        event_list.append(log_text)    


    # Stop at Non-service Floor (原来叫作 Over Travel)     
    if 0.3 < position_dev < 7 and seq['Velocity'].iloc[-1] < 0.5 and seq['Motor_KF'].iloc[-1] < 10:
        log_text = {
            "time": str(end_time),
            "status ID": 2.7,
            "event": "Stop at Non-service Floor",                    
            "description": "Lift stops at the level not at the proper position (>+/- 300mm)",
            "floor": str(Arrive_F),
            "delsign":0  
                }
        event_list.append(log_text)  

    # Sudden Stop     
    # if 0.5 < seq['Velocity'].iloc[-10:].median() < 3 and seq['Motor_KF'].iloc[-1] > 10:  #用该trip的最后1s速度的中位数判断以及最后一个Motor Current值判断    
    #用该trip的最后0.5s速度的中位数或者距离偏移 判断以及最后一个Motor Current值判断   
    if seq['Motor_KF'].iloc[-1] > 10 and (0.5 < seq['Velocity'].iloc[-10:].median() < 3 or 0.3 < position_dev < 7):        
        log_text = {
            "time": str(end_time),
            "status ID": 2.8,
            "event": "Sudden stop",                    
            "description": "Lift suddenly stops at the mid of travelling ",
            "floor": str(Arrive_F),
            "delsign":0                 
            }
        event_list.append(log_text)  

    # Start failure     # update on 2023-3-20, start failure 在door carseg那边判断 
    # if mileage < 0.2 and ~np.isnan(Arrive_F) and Resv3I_run > paras['thres_resv3']: # (1)没移动；（2）Arrive_F 有值; (3) Resv3 是处于电梯运行的level
    
    #     log_text = {
    #         "time": str(end_time),
    #         "status ID": 2.9,
    #         "event": "Start failure",                    
    #         "description": "Lift cannot start successfully ",
    #         "floor": str(Arrive_F),
    #         "delsign":0  
    #             }
    #     event_list.append(log_text)  


         
    CarStat_text = {
        'start_time':start_time,
        'end_time':end_time,
        'duration':duration,
        'hour':hour,
        'mileage':round(mileage,2),
        'Depart_F':Depart_F,
        'Arrive_F':Arrive_F,
        'Fs_travelled':Fs_travelled,
        'F_Travelled':F_Travelled,
        'Dir_Travelled':Dir_Travelled,
        'MotorI_start':round(MotorI_start,2), # Full Field current
        'MotorI_peak':round(MotorI_peak,2),   # Run Field current
        'MotorI_brake':np.nan, 
        'MotorI_steady':np.nan,         
        'BrakeI_steady':round(BrakeI_steady,2),
        'BrakeI_peak':round(BrakeI_peak,2),
        'SafetyI_run':round(SafetyI_run,2),
        'Resv1I_start':round(Resv1I_start,2), #Armature Starting current
        'Resv1I_peak':round(Resv1I_peak,2), #Armature Peak current
        'Resv1I_brake':round(Resv1I_brake,2),  #Armature Brake current
        'Resv1I_steady':round(Resv1I_steady,2), #Armature Steady current          
        'Resv2I_start':np.nan,
        'Resv2I_peak':np.nan,
        'Resv2I_brake':np.nan, 
        'Resv2I_steady':np.nan,        
        'Resv3I_run':round(Resv3I_run,2),
        'Speed_peak':round(Speed_peak,2)
        } 
    
    CarStat_text = {k: str(CarStat_text[k]) if pd.isnull(CarStat_text[k]) else CarStat_text[k] for k in CarStat_text }
            
    return CarStat_text, event_list



def Stats_PerDoorSeg(seq, paras, event_list): # Statistic information for each car segment/cycle
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
        if paras['DoorWaveform_type'] == 1: # 一个波峰》一对
            
            # num_Door = len(np.argwhere(np.diff(np.sign(paras['line_Door']-seq['Door_KF']))))/4 # number of door closing and opening pairs
            #num_Door = ceil(num_Door)
            peaks, _ = find_peaks(seq['Door_KF'], height = paras['line_Door'])
            num_Door = ceil(len(peaks))  # pairs of door open&closes

        elif paras['DoorWaveform_type'] == 2: # 两个波峰》一对
            peaks, _ = find_peaks(seq['Door_KF'], height = paras['line_Door'],distance = 50)
            num_Door = ceil(len(peaks)/2)  # pairs of door open&closes

            
        elif paras['DoorWaveform_type'] == 4: # 四个波峰》一对
            peaks, _ = find_peaks(seq['Door_KF'], height = paras['line_Door'])
            num_Door = ceil(len(peaks)/4)  # pairs of door open&closes
            
            
        elif paras['DoorWaveform_type'] == 0: # 是矩形pattern

            num_Door = len(np.argwhere(np.diff(np.sign(paras['line_Door']-seq['Door_KF']))))/4 # number of door closing and opening pairs
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

    ######  Update on 2022-12-19  #####      
    # if paras['RMU_ID']==14:
    #     seq_dooropen = seq.loc[(seq['Safety']>=paras['Dooropen_SafetyI_range'][0]) & (seq['Safety']<=paras['Dooropen_SafetyI_range'][1])]
    #     DoorOpen_Duration1 = len(seq_dooropen)/20
    #     DoorOpen_Duration2 = len(seq.loc[seq['Resv-3'] < paras['thres_resv3']])/20
    #     DoorOpen_Duration = max(DoorOpen_Duration1,DoorOpen_Duration2)
            
    # else:       
    #     if pd.isnull(paras['thres_resv3']):
    #         seq_dooropen = seq.loc[(seq['Safety']>=paras['Dooropen_SafetyI_range'][0]) & (seq['Safety']<=paras['Dooropen_SafetyI_range'][1])]
    #         DoorOpen_Duration = len(seq_dooropen)/20
    #     else:
    #         DoorOpen_Duration = len(seq.loc[seq['Resv-3'] < paras['thres_resv3']])/20 # door open duration within each cycle    

    ############################################################
    ############  3. Event Detection based on Rules ############
    ###################################s########################
    
    if np.isnan(paras['Floor']).all() or np.isnan(seq['Distance']).all():        
        Stop_F=np.nan
    else:
        Stop_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor'])) - 1 # STOP floor of this cycle
    
    # UCM     
    if distance > 100.3 and DoorI_peak > paras['line_Door']:
        log_text = {
            "time": str(end_time),
            "status ID": 2.1,
            "event": "UCM",
            "description": "lift car has travelled beyond door zone (+/- 300 mm of floor level)",
            "floor": str(Stop_F),
            "delsign":0  
                }
        event_list.append(log_text)     


    # if paras['DoorOpenLong_FLAG'] == 1 and 30 < DoorOpen_Duration < 55 and Stop_F != 0: 
    if paras['DoorOpenLong_FLAG'] == 1 and DoorOpen_Duration > paras['thres_DoorOpenDuration'] and Stop_F != 0: 
        
        if Stop_F == -1:
            Stop_F = 'LG'
            
        if paras['RMU_ID'] != 14: # disable the long door openning  duration alert for QML3 (update on 2023-02-20)
            
            log_text = {
                "time": str(end_time),
                "status ID": 3.1,
                "event": "Door anomaly",
                "description": "Door openning time lasts longer than " + str(round(DoorOpen_Duration,2)) + " s at " + str(Stop_F) + "/F",
                "floor": str(Stop_F),
                "delsign":0  
                    }
            event_list.append(log_text)   
        
    if DoorI_peak < paras['DrIpeak_Range'][0] or DoorI_peak > paras['DrIpeak_Range'][1]: #2023-2-22 从非AI model里提取出来
        
        log_text = {
            "time": str(end_time),
            "status ID": 3.1,
            "event": "Door anomaly",
            "description": "Anomaly door motor current magnitude:" + str(round(DoorI_peak,2)) + " A",
            "floor": str(Stop_F),
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
            "floor": str(Stop_F),
            "delsign":0                  
                }
        event_list.append(log_text)    



    #### 检测是否有releveling case 发生 ####           
    Relvl_list = []

    seq['carseg_flag'] = np.sign(seq['Brake_KF']-0.01) # -1 indicates brake close and 1 indicates brake open
    
    carseg_group = seq[seq['carseg_flag'] == 1].groupby((seq['carseg_flag'] != 1).cumsum())
    
    for k, v in carseg_group:
        
        Relvl_list.append(v)  # Relevling cycle list

    # 删除Releveling segment中小于1s的cycle        
    Relvl_list = list(filter(lambda x: len(x)>20, Relvl_list))            

    # if len(Relvl_list)>0 and (num_Door == 0 or num_Door == np.nan): # start failure
    if len(Relvl_list)>0 and seq['Brake_KF'].iloc[0] < 0.2 and (num_Door == 0 or num_Door == np.nan): # start failure，update on 2023-1-11, add one more condition for special case such as QM and QE lift
        
        log_text = {
            "time": str(end_time),
            "status ID": 2.9,
            "event": "Start failure",                    
            "description": "Lift cannot start successfully ",
            "floor": str(Stop_F),
            "delsign":0              
                }
        event_list.append(log_text)          
        
    ######  Update on 2023-2-10  #####      
    if paras['RMU_ID']==14: # 对于QML3 需要通过判断电梯停止时 motor电流fluctuate的时间/频率来判断门是存在故障
        # seq['Motor'][seq['Motor'] > 1].count() 
        seq['MotorWave_flag'] = np.sign(seq['Motor']-1) 
        MotorWave_count = len(seq[seq['MotorWave_flag'] == 1].groupby((seq['MotorWave_flag'] != 1).cumsum())) # motor 电流fluctuate的次数   
        if MotorWave_count > 70:
            
            if Stop_F == -1:
                Stop_F = 'LG'
            elif Stop_F == 0:
                Stop_F = 'G'
                
            log_text = {
                "time": str(end_time),
                "status ID": 3.1,
                "event": "Door anomaly",
                "description": "Irregular door pattern identified at " + str(Stop_F) + "/F",
                "floor": str(Stop_F),
                "delsign":0  
                    }
            event_list.append(log_text)  
            

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


# def Door_nos(seq,end_time,thres_numDoor,line_Door,event_list):
#     num_Door = len(np.argwhere(np.diff(np.sign(line_Door-seq['Door_KF']))))/2 # number of door closing and opening times
    
#     if num_Door >= thres_numDoor:
        
#         log_text = {
#             "time": str(end_time),
#             "status ID": 3.1,
#             "event": "Door anomaly",
#             "description": "Door anomaly open & close",
#                 }
#         event_list.append(log_text)
        
#     # pairs_Door = round(num_Door/2) # number of door closing and opening pairs
#     return event_list

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
    with tf.Graph().as_default():    
        with tf.Session() as sess:    
            K.set_session(sess)
            
            if model_name == 'EMSD_V1':
                model = model_from_yaml(open('/app/hkdtspts/devpy/model.yaml').read(), custom_objects={"AttentionLSTM": AttentionLSTM})
                model.load_weights('/app/hkdtspts/devpy/my_model_weights.h5', by_name=False)
                # model = model_from_yaml(open('./devpy/model.yaml').read(), custom_objects={"AttentionLSTM": AttentionLSTM})
                # model.load_weights('./devpy/my_model_weights.h5', by_name=False)    
            else:
                model = load_model('/app/hkdtspts/devpy/'+ model_name,custom_objects={"AttentionLSTM": AttentionLSTM}, compile=False)                        
                # model = load_model('./devpy/'+ model_name,custom_objects={"AttentionLSTM": AttentionLSTM}, compile=False)        

            
            X_mat = np.array(seq[['Door_KF']]) 
            ##### Update on 2022-12-30 22:44:00 ####
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
            "status ID": 3.1,
            "event": "Door anomaly",                    
            "description": "Door anomaly open & close",
            "floor": Stop_F,
            "delsign":0  
                }
        event_list.append(log_text)  
    return event_list


########### Update on 2023-02-27 22:34:00 暂时不判断Brake ######################

# def LSTMAE_Brake(criterion,brake_result,brake_list,CarSeg_list,BrakeError_max,event_list):
#     brake_list = pd.DataFrame(brake_list)
#     # motor_list = pd.DataFrame(motor_list)
#     brake_data = pre_data(brake_list)
#     # motor_data = pre_data(motor_list)
    
#     f_brake = open('/app/hkdtspts/devpy/Brake_LSTM_AE.pt', 'rb')
#     # f_brake = open('./devpy/Brake_LSTM_AE.pt', 'rb')
    
#     # f_motor = open('/app/hkdtspts/devpy/Motor_LSTM_AE.pt', 'rb')
#     lstm_ae_brake = torch.load(f_brake, map_location='cpu')
#     lstm_ae_brake = lstm_ae_brake.cpu()
#     lstm_ae_brake.eval()
#     # lstm_ae_motor = torch.load(f_motor, map_location='cpu')
#     # lstm_ae_motor = lstm_ae_motor.cpu()
#     # lstm_ae_motor.eval()
#     yBrake_pred = lstm_ae_brake(brake_data).squeeze()
#     # yMotor_pred = lstm_ae_motor(motor_data).squeeze()

#     for i in range(len(CarSeg_list)):

#         loss_brake = criterion(brake_data[i], yBrake_pred[i]).detach().numpy()
#         # loss_motor = criterion(motor_data[i], yMotor_pred[i]).detach().numpy()

        
#         if loss_brake >= BrakeError_max:
#             brake_result.append(1)
#         else:
#             brake_result.append(0)
            
#         # if loss_motor >= MotorError_max:
#         #     motor_result.append(1)
#         # else:
#         #     motor_result.append(0)


            
#     for i, j in enumerate(brake_result):
#         if j == 1:
#             seq = CarSeg_list[i]
#             end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
#             log_text = {
#                             "time": str(end_time),
#                             "status ID": 3.2,
#                             "event": "Brake unsmooth operation",
#                             "description": "Brake unsmooth operation",
#                             "floor": 'nan',
#                             "delsign":0  
#                                 }
#             event_list.append(log_text) # 把brake异常存到event_list里
            
#     return event_list
    

## 执行数据分析和AI主程序

def do_action(dataset, paras, CarSeg_list, DoorSeg_list):
    event_list = []
    CarStat_list = []
    DoorStat_list = []
       
    # AI_FLAG = paras['AI_FLAG']
    Motor_type = paras['Motor_type']

    ##计算loss
    criterion = torch.nn.MSELoss(reduction='sum')
    # criterion2 = torch.nn.L1Loss(reduction='sum')
    ############################################################
    ################  1. CarSeg_list ########################
    ############################################################
    if len(CarSeg_list) > 0:
        
        brake_list = []
        brake_result = []
        # motor_list = []
        # motor_result = []
        for i in range(len(CarSeg_list)):
            seq = CarSeg_list[i]
            end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            
            ####################### Car Rules #########################
            if Motor_type == 'AC':
                CarStat_text,event_list = Stats_PerCarSeg_AC(seq,paras,event_list)
            elif Motor_type == 'DC':
                CarStat_text,event_list = Stats_PerCarSeg_DC(seq,paras,event_list)

            CarStat_list.append(CarStat_text) 

            ####################### Motor AI Model #########################

            if paras['AI_Motor_Model'] != '': # Trigger the  Motor AI module 
            

                seq.loc[seq['Motor'] < 0, 'Motor'] = 0
                model_name = paras['AI_Motor_Model']
                event_list = LSTMFCN_Motor(seq,end_time,model_name,event_list)
    
                ########### Update on 2022-12-31 19:12:00  ######################
                # seq.loc[seq['Brake'] < 0, 'Brake'] = 0
                # seq_brake = list(np.reshape(np.array(seq['Brake']), (1, -1)))
                # brake_list = brake_list + seq_brake
                # BrakeError_max = paras['BrakeError_max']
                # event_list = LSTMAE_Brake(criterion,brake_result,brake_list,CarSeg_list,BrakeError_max,event_list)
                ########### ########### ########### ########### ########### ###########
    else:
        end_time = dataset.index[-1].strftime('%Y-%m-%d %H:%M:%S.%f')

                 

    ############################################################
    ################  2. DoorSeg_list ########################
    ############################################################
    if len(DoorSeg_list) > 0:

        # door_list = []
        # door_result = []
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




def LockMode(dataset,paras):
    event_list = []
    seq_lock = dataset.loc[(dataset['Safety']>=paras['Locked_SafetyI_range'][0]) & (dataset['Safety']<=paras['Locked_SafetyI_range'][1])]    

    if len(seq_lock)>=1000:
        log_text = {
            "time": str(seq_lock.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 5,
            "event": "Locked",
            "description": "Lock mode - out of service",
            "floor": 'nan',
            "delsign":0              
                            }
        event_list.append(log_text) # 把锁机事件存到event_list里   
    return event_list
        
def SafetyTrip(dataset,paras): 
    event_list = []
    seq_trip = dataset.loc[(dataset['Safety']>=paras['Tripped_SafetyI_range'][0]) & (dataset['Safety']<=paras['Tripped_SafetyI_range'][1])]

    if len(seq_trip)>=120:        
        log_text = {
            "time": str(seq_trip.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 2.6,
            "event": "Safety tripped (Idle)",
            "description": "Urgent - Safety tripped when lift is in idle",
            "floor": 'nan',
            "delsign":0  
                }
        event_list.append(log_text) # 把trip事件存到event_list里  
    return event_list

def SafetyTrip2(dataset,paras): # safety 和 door 同时满足某个条件
    event_list = []
    seq_trip1 = dataset.loc[(dataset['Safety']>=paras['Tripped_SafetyI_range'][0]) & (dataset['Safety']<=paras['Tripped_SafetyI_range'][1])]
    seq_trip2 = dataset.loc[(dataset['Door']>=paras['Tripped_SafetyI_range'][2]) & (dataset['Door']<=paras['Tripped_SafetyI_range'][3])]

    if (len(seq_trip1)>=120) & (len(seq_trip2)>=120):        
        log_text = {
            "time": str(seq_trip1.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 2.6,
            "event": "Safety tripped (Idle)",
            "description": "Urgent - Safety tripped when lift is in idle",
            "floor": 'nan',
            "delsign":0  
                }
        event_list.append(log_text) # 把trip事件存到event_list里  
    return event_list

def SafetyTrip3(dataset,paras): # safety 连续trip 大于3s (# update on 2023-3-20)
    event_list = []
    TripSeg_list = []

    dataset['trip_flag'] = np.sign(dataset['Safety']-paras['Tripped_SafetyI_range'][1]) # -1 indicates safety trip and 1 indicates safety not trip


    tripseg_group = dataset[dataset['trip_flag'] == -1].groupby((dataset['trip_flag'] != -1).cumsum())
    
    for k, v in tripseg_group:
        
        TripSeg_list.append(v)  # Trip list
    
    TripSeg_list = list(filter(lambda x: len(x)>60, TripSeg_list)) #保留trip 时间大于3s的seg
           
    if len(TripSeg_list)>0:        
        log_text = {
            "time": str(TripSeg_list[-1].index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 2.6,
            "event": "Safety tripped (Idle)",
            "description": "Urgent - Safety tripped when lift is in idle",
            "floor": 'nan',
            "delsign":0  
                }
        event_list.append(log_text) # 把trip事件存到event_list里  
    return event_list

def SafetyInspection(dataset,paras): # 用safety circuit电流判断inspection事件
    event_list = []
    seq_inspection_run = dataset.loc[(dataset['Safety']>=paras['InspectionRun_range'][0]) & (dataset['Safety']<=paras['InspectionRun_range'][1])]
    seq_inspection_stop = dataset.loc[(dataset['Safety']>=paras['InspectionStop_range'][0]) & (dataset['Safety']<=paras['InspectionStop_range'][1])]
        
    if len(seq_inspection_run)>=40:
        log_text = {
            "time": str(seq_inspection_run.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 4,
            "event": "Inspection mode",
            "description": "Inspection mode - maintenance run",
            "floor": 'nan',
            "delsign":0  
                            }
        event_list.append(log_text) # 把检修事件存到event_list里   
        
    elif len(seq_inspection_stop)>=40:
        log_text = {
            "time": str(seq_inspection_stop.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 4,
            "event": "Inspection mode",
            "description": "Inspection mode - maintenance stop",
            "floor": 'nan',
            "delsign":0  
                            }
        event_list.append(log_text) # 把检修事件存到event_list里  
    return event_list

            
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


## 序列切割函数，返回 CarSeg_list, DoorSeg_list,其中DOOR由resv3 提取的。
