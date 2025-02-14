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
def Data_segment(dataset):

    dataset = dataset.reset_index()
    dataset = dataset.rename(columns={'index': 'Time'})
    # use Kalman Filtering to smooth the data 
    smoother = KalmanSmoother(component='level_trend',
                          component_noise={'level': 0.1, 'trend': 0.1})
    

    dataset.loc[dataset['Distance'] >= 180,'Distance'] = np.nan # remove the outliers
    dataset.loc[dataset['Distance'] == 0,'Distance'] = np.nan # remove the outliers

    dataset['Velocity'] = np.abs(dataset['Distance'].diff() / 0.05)  # calculate the abs velocity
   
    dataset = dataset.interpolate() # 用插值法补全缺失数据  
    
    dataset.loc[dataset['Motor'] < 0.15,'Motor'] = 0
    dataset.loc[dataset['Brake'] < 0.15,'Brake'] = 0
    dataset.loc[dataset['Door'] < 0.1,'Door'] = 0
    dataset.loc[dataset['Safety'] < 0,'Safety'] = 0
    dataset.loc[dataset['Resv-1'] < 0.1,'Resv-1'] = 0
    dataset.loc[dataset['Resv-2'] < 0.1,'Resv-2'] = 0
    dataset.loc[dataset['Resv-3'] < 0,'Resv-3'] = 0
    dataset.loc[dataset['Distance'] < 0,'Distance'] = 0
    dataset.loc[dataset['Velocity'] < 0,'Velocity'] = 0
    
    values = dataset[['Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','Distance','Velocity']]
    smooth_seq = smoother.smooth(values.T).smooth_data.T # all smoothed sequences, if nan for a whole column, it will be 0.   
    dataset[['Motor_KF','Brake_KF','Safety_KF','Door_KF','Resv1_KF','Resv2_KF','Resv3_KF','Dist_KF','Vel_KF']] = smooth_seq
    
    dataset['Motor_KF'] = dataset['Motor_KF'].apply(lambda x: np.nan if dataset['Motor'].isnull().all() else x)
    dataset['Brake_KF'] = dataset['Brake_KF'].apply(lambda x: np.nan if dataset['Brake'].isnull().all() else x)
    dataset['Safety_KF'] = dataset['Safety_KF'].apply(lambda x: np.nan if dataset['Safety'].isnull().all() else x)
    dataset['Door_KF'] = dataset['Door_KF'].apply(lambda x: np.nan if dataset['Door'].isnull().all() else x)
    dataset['Resv1_KF'] = dataset['Resv1_KF'].apply(lambda x: np.nan if dataset['Resv-1'].isnull().all() else x)        
    dataset['Resv2_KF'] = dataset['Resv2_KF'].apply(lambda x: np.nan if dataset['Resv-2'].isnull().all() else x)
    dataset['Resv3_KF'] = dataset['Resv3_KF'].apply(lambda x: np.nan if dataset['Resv-3'].isnull().all() else x)
    dataset['Dist_KF'] = dataset['Dist_KF'].apply(lambda x: np.nan if dataset['Distance'].isnull().all() else x)
    dataset['Vel_KF'] = dataset['Vel_KF'].apply(lambda x: np.nan if dataset['Velocity'].isnull().all() else x)

    
    dataset.loc[dataset['Motor_KF'] < 0,'Motor_KF'] = 0
    dataset.loc[dataset['Brake_KF'] < 0.15,'Brake_KF'] = 0
    dataset.loc[dataset['Safety_KF'] < 0,'Safety_KF'] = 0
    dataset.loc[dataset['Door_KF'] < 0,'Door_KF'] = 0
    dataset.loc[dataset['Resv1_KF'] < 0,'Resv1_KF'] = 0
    dataset.loc[dataset['Resv2_KF'] < 0,'Resv2_KF'] = 0
    dataset.loc[dataset['Resv3_KF'] < 0,'Resv3_KF'] = 0
    dataset.loc[dataset['Dist_KF'] < 0,'Dist_KF'] = 0

    CarSeg_list = []
    DoorSeg_list = []
    ind_start = 0
    ind_end = 0
    ind_prestart = 0
    
    
    i = 1
    while i < (len(dataset)-1): # i < (len(dataset) - 120):
    
    
        if (dataset.loc[i, 'Brake_KF'] == 0) & (dataset.loc[(i + 1):(i + 120), 'Brake_KF'] > 0).all():
            #        seg_ind[i-10,0] = i # cycle starting index
            dataset.loc[i, 'seg_flag'] = 1  # use 1 to mark cycle starting point
            ind_start = i
    
            DoorSeg_list.append(dataset.iloc[ind_end:ind_start])  # Door motion cycle list
    
    
        elif (dataset.loc[(i - 120):(i - 1), 'Brake_KF'] > 0).all() & (dataset.loc[i, 'Brake_KF'] == 0):
    
            dataset.loc[i+1, 'seg_flag'] = 2  # use 2 to mark cycle ending point
            ind_end = i+1
    
            CarSeg_list.append(dataset.iloc[ind_start:ind_end])  # lift car traveling cycle list
    
            ind_prestart = ind_start
    
        i += 1
    
    if ind_start == ind_prestart:
        DoorSeg_list.append(dataset.iloc[ind_end:i])  # Door motion cycle list
        
        
    # 删除第一个不完整的car cycle
    if len(CarSeg_list)>0:
        if CarSeg_list[0].iloc[0]['Brake_KF']>0:
    
            del (CarSeg_list[0])
    # 删除car segment中小于6s的cycle        
    CarSeg_list = list(filter(lambda x: len(x)>120, CarSeg_list))
    # 删除door segment中为空的cycle        
    DoorSeg_list = list(filter(lambda x: len(x)>0, DoorSeg_list))
    # 删除door segment中首尾segment     
    if len(DoorSeg_list)>0:
     
        if (DoorSeg_list[0].iloc[0]['Time'] == dataset['Time'].iloc[0]) & (DoorSeg_list[0].iloc[-1]['Time'] != dataset['Time'].iloc[-2]):
            del (DoorSeg_list[0])
                 
    if len(DoorSeg_list)>0:
  
        if (DoorSeg_list[-1].iloc[-1]['Time'] == dataset['Time'].iloc[-2]) & (len(DoorSeg_list[-1])<400): # 末尾小于20s的删除
            del (DoorSeg_list[-1])            


    
    # DoorSeg_list = list(filter(lambda x: len(x)>140, DoorSeg_list))
    
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

def Stats_PerCarSeg(seq, paras, event_list): # Statistic information for each car segment/cycle

    ############################################################
    ################  1. Traffic Information ###################
    ############################################################
    start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
    duration = (seq.iloc[-1]['Time'] - seq.iloc[0]['Time']).total_seconds()
    hour = seq.iloc[-1]['Time'].hour                    
    mileage = np.abs(seq['Distance'].iloc[-1] - seq['Distance'].iloc[0])
    if np.isnan(paras['Floor']).all():        
        Depart_F=Arrive_F=np.nan
    else:
        Depart_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor']))  # start floor of this cycle
        Arrive_F = np.nanargmin(np.abs(seq['Distance'].iloc[-1] - paras['Floor']))  # end floor of this cycle        
    Fs_travelled = Arrive_F - Depart_F # the floors the elevator travels, + upward, - downward
    F_Travelled = np.abs(Fs_travelled) # the floors the elevator travels
    Dir_Travelled = np.sign(Fs_travelled) # +1 upward, -1 downward, 0 releveling
    
    
    ############################################################
    ################  2. Key Parameters Calculation ############
    ############################################################
    MotorI_start = seq['Motor_KF'].iloc[25:75].mean()  # starting current of motor
    MotorI_peak = seq['Motor_KF'].max()  # peak current of motor
    
    BrakeI_steady = seq['Brake_KF'].iloc[36:-4].mean() #average brake steady current when car runs
    BrakeI_peak = seq['Brake_KF'].max() # peak brake current when car runs 
    
    SafetyI_run = seq['Safety'].mean() #average safety current when car runs

    Resv1I_start = seq['Resv1_KF'].iloc[25:75].mean()  # starting current of resv1
    Resv1I_peak = seq['Resv1_KF'].max()  # peak current of resv1

    Resv2I_start = seq['Resv2_KF'].iloc[25:75].mean()  # starting current of resv2
    Resv2I_peak = seq['Resv2_KF'].max()  # peak current of resv2

    Resv3I_run = seq['Resv-3'].mean() # average resv3 current when car runs
        
    Speed_peak = seq['Velocity'].max() # peak speed  when car runs
    
    
    ############################################################
    ############  3. Event Detection based on Rules ############
    ############################################################
    # if (MotorI_peak > paras['MotIpeak_Max']) | (MotorI_peak < paras['MotIpeak_Min']): # out of the normal motor current range
    if not paras['MotIpeak_Range'][0] < MotorI_peak < paras['MotIpeak_Range'][1]: # out of the normal motor peak current range
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly motor peak current magnitude:" + str(round(MotorI_peak,2)),
                 }
         event_list.append(log_text) 
         
    # if (BrakeI_steady > paras['BrIsteady_Max']) | (BrakeI_steady < paras['BrIsteady_Min']): # out of the normal brake current range
    if not paras['BrIsteady_Range'][0] < BrakeI_steady < paras['BrIsteady_Range'][1]: # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
              "description": "anomaly brake steady current magnitude:" + str(round(BrakeI_steady,2)),
                 }
         event_list.append(log_text)

    # if (BrakeI_peak > paras['BrIpeak_Max']) | (BrakeI_peak < paras['BrIpeak_Min']): # out of the normal brake current range
    if not paras['BrIpeak_Range'][0] < BrakeI_peak < paras['BrIpeak_Range'][1]: # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
             "description": "anomaly brake peak current magnitude:" + str(round(BrakeI_peak,2)),
                 }
         event_list.append(log_text)       
 
    # ACO     
    if Dir_Travelled == 1:  # 判断电梯是否处于上升状态

        seq_overspeed = seq.loc[seq['Velocity'] > 1.15 * paras['RatedSpeed']] # When lift car is travelling upward and >115% of rated speed

        if len(seq_overspeed) > 80: # overspeed 超过4s
            log_text = {
                "time": str(end_time),
                "status ID": 2.2,
                "event": "ACO",
                 "description": "Lift ascending over speed ",
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
        'BrakeI_steady':round(BrakeI_steady,2),
        'BrakeI_peak':round(BrakeI_peak,2),
        'SafetyI_run':round(SafetyI_run,2),
        'Resv1I_start':round(Resv1I_start,2),
        'Resv1I_peak':round(Resv1I_peak,2),
        'Resv2I_start':round(Resv2I_start,2),
        'Resv2I_peak':round(Resv2I_peak,2),
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
        num_Door = len(np.argwhere(np.diff(np.sign(paras['line_Door']-seq['Door_KF']))))/2 # number of door closing and opening times
        num_Door = ceil(num_Door)
    else:
        num_Door = np.nan
        
    if num_Door > 0:
        DoorI_peak = seq['Door_KF'].max()
    else:
        DoorI_peak = np.nan
        
    # if not np.isnan(paras['thres_resv3']):
    #     DoorOpen_Duration = len(seq.loc[seq['Resv-3'] < paras['thres_resv3']])/20 # door open duration within each cycle    
    # else:
    #     DoorOpen_Duration = np.nan
        
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

    if DoorOpen_Duration > 25: # 开门时间过长
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
                "description": "Anomaly door motor current magnitude",
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
        } 
    
    DoorStat_text = {k: str(DoorStat_text[k]) if pd.isnull(DoorStat_text[k]) else DoorStat_text[k] for k in DoorStat_text }

    return DoorStat_text, event_list


def Door_nos(seq,end_time,thres_numDoor,line_Door,event_list):
    num_Door = len(np.argwhere(np.diff(np.sign(line_Door-seq['Door_KF']))))/2 # number of door closing and opening times
    
    if num_Door >= thres_numDoor:
        
        log_text = {
            "time": str(end_time),
            "status ID": 3.1,
            "event": "Door anomaly",
            "description": "Door anomaly open & close",
                }
        event_list.append(log_text)
        
    # pairs_Door = round(num_Door/2) # number of door closing and opening pairs
    return event_list

def LSTMFCN_Motor(seq,end_time,event_list):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            K.set_session(sess)
            model = model_from_yaml(open('./devpy/inspection_model.yaml').read(), custom_objects={"AttentionLSTM": AttentionLSTM})
            model.load_weights('./devpy/inspection_model_weights.h5', by_name=False)
            X_mat = np.array(seq[['Motor']])
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
                }
        event_list.append(log_text)  
    return event_list

def LSTMFCN_Door(seq,end_time,event_list):
    with tf.Graph().as_default():
    
        with tf.Session() as sess:
    
            K.set_session(sess)
            
            model = model_from_yaml(open('./devpy/model.yaml').read(), custom_objects={"AttentionLSTM": AttentionLSTM})
            model.load_weights('./devpy/my_model_weights.h5', by_name=False)
            X_mat = np.array(seq[['Door_KF']])
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
                }
        event_list.append(log_text)  
    return event_list

def LSTMAE_Brake(criterion,brake_result,brake_list,CarSeg_list,BrakeError_max,event_list):
    brake_list = pd.DataFrame(brake_list)
    # motor_list = pd.DataFrame(motor_list)
    brake_data = pre_data(brake_list)
    # motor_data = pre_data(motor_list)
    f_brake = open('./devpy/Brake_LSTM_AE.pt', 'rb')
    # f_motor = open('./devpy/Motor_LSTM_AE.pt', 'rb')
    lstm_ae_brake = torch.load(f_brake, map_location='cpu')
    lstm_ae_brake = lstm_ae_brake.cpu()
    lstm_ae_brake.eval()
    # lstm_ae_motor = torch.load(f_motor, map_location='cpu')
    # lstm_ae_motor = lstm_ae_motor.cpu()
    # lstm_ae_motor.eval()
    yBrake_pred = lstm_ae_brake(brake_data).squeeze()
    # yMotor_pred = lstm_ae_motor(motor_data).squeeze()

    for i in range(len(CarSeg_list)):

        loss_brake = criterion(brake_data[i], yBrake_pred[i]).detach().numpy()
        # loss_motor = criterion(motor_data[i], yMotor_pred[i]).detach().numpy()

        
        if loss_brake >= BrakeError_max:
            brake_result.append(1)
        else:
            brake_result.append(0)
            
        # if loss_motor >= MotorError_max:
        #     motor_result.append(1)
        # else:
        #     motor_result.append(0)


            
    for i, j in enumerate(brake_result):
        if j == 1:
            seq = CarSeg_list[i]
            end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            log_text = {
                            "time": str(end_time),
                            "status ID": 3.2,
                            "event": "Brake unsmooth operation",
                            "description": "Brake unsmooth operation",
                                }
            event_list.append(log_text) # 把brake异常存到event_list里
            
    return event_list
    

## 执行数据分析和AI主程序

def do_action(dataset, paras, CarSeg_list, DoorSeg_list):
    event_list = []
    CarStat_list = []
    DoorStat_list = []
       
    AI_FLAG = paras['AI_FLAG']
    # Floor = paras['Floor']

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
            
            ####################### Rules #########################
            CarStat_text,event_list = Stats_PerCarSeg(seq,paras,event_list)
            # event_list = ACO(seq,end_time,RatedSpeed,event_list)
            # event_list, BrakeI_steady, BrakeI_max = Brake_mag(seq,end_time,BrIsteady_Max,BrIsteady_Min,BrIpeak_Max,BrIpeak_Min,event_list)
            # event_list, MotorI_start, MotorI_max = Motor_mag(seq,end_time,MotIpeak_Max,MotIpeak_Min,event_list)

            CarStat_list.append(CarStat_text) 

            if AI_FLAG == 1: # Trigger the AI module

                seq.loc[seq['Motor'] < 0, 'Motor'] = 0
                event_list = LSTMFCN_Motor(seq,end_time,event_list)
    
                
                seq.loc[seq['Brake'] < 0, 'Brake'] = 0
                seq_brake = list(np.reshape(np.array(seq['Brake']), (1, -1)))
                brake_list = brake_list + seq_brake
                BrakeError_max = paras['BrakeError_max']
                event_list = LSTMAE_Brake(criterion,brake_result,brake_list,CarSeg_list,BrakeError_max,event_list)
    

    ############################################################
    ################  2. DoorSeg_list ########################
    ############################################################

    # door_list = []
    # door_result = []
    for i in range(len(DoorSeg_list)):
        seq = DoorSeg_list[i]        
        end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
         ####################### Rules #########################
        
        DoorStat_text,event_list = Stats_PerDoorSeg(seq,paras,event_list)

        DoorStat_list.append(DoorStat_text) 
        
        if AI_FLAG == 1: # Trigger the AI module
                        
            event_list = LSTMFCN_Door(seq,end_time,event_list)
            

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
                            }
        event_list.append(log_text) # 把锁机事件存到event_list里   
    return event_list
        
def SafetyTrip(dataset,paras): 
    event_list = []
    seq_trip = dataset.loc[(dataset['Safety']>=paras['Tripped_SafetyI_range'][0]) & (dataset['Safety']<=paras['Tripped_SafetyI_range'][1])]

    if len(seq_trip)>=40:        
        log_text = {
            "time": str(seq_trip.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 2.6,
            "event": "Safety tripped (Idle)",
            "description": "Urgent - Safety tripped when lift is in idle)",
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
                            }
        event_list.append(log_text) # 把检修事件存到event_list里   
        
    elif len(seq_inspection_stop)>=40:
        log_text = {
            "time": str(seq_inspection_stop.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 4,
            "event": "Inspection mode",
            "description": "Inspection mode - maintenance stop",
                            }
        event_list.append(log_text) # 把检修事件存到event_list里  
    return event_list

            
def DailyParas_Calculate(dataset,paras): # dataset为1s sample!!! 计算日参数，包括runtime,idletime,door open time,lock time,inspection time, rmu offline time, safety trip time, voltage dip time.
    Date = dataset.index[0].strftime('%Y-%m-%d')

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
