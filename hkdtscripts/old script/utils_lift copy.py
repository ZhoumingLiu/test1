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

from keras.models import load_model
from utils.layer_utils import AttentionLSTM
from keras.models import model_from_yaml

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
    
    if dataset['Distance'].isnull().all():
        dataset = dataset.interpolate() # 用插值法补全缺失数据     
        dataset.loc[dataset['Brake'] < 0.15,'Brake'] = 0
        dataset.loc[dataset['Door'] < 0.1,'Door'] = 0
        dataset.loc[dataset['Motor'] < 0.1,'Motor'] = 0
        smoother.smooth(dataset[['Motor', 'Brake', 'Door', 'Resv-3']].T)
        new_seq = smoother.smooth_data.T
        dataset['Motor_KF'] = new_seq[:, 0]
        dataset['Brake_KF'] = new_seq[:, 1]
        dataset['Door_KF'] = new_seq[:, 2]
        dataset['Resv3_KF'] = new_seq[:, 3]
    else:
        dataset.loc[dataset['Distance'] >= 180,'Distance'] = np.nan
        dataset = dataset.interpolate() # 用插值法补全缺失数据     
        dataset.loc[dataset['Brake'] < 0.15,'Brake'] = 0
        dataset.loc[dataset['Door'] < 0.1,'Door'] = 0
        dataset.loc[dataset['Motor'] < 0.1,'Motor'] = 0
        smoother.smooth(dataset[['Motor', 'Brake', 'Door', 'Resv-3','Distance']].T)
        new_seq = smoother.smooth_data.T
        dataset['Motor_KF'] = new_seq[:, 0]
        dataset['Brake_KF'] = new_seq[:, 1]
        dataset['Door_KF'] = new_seq[:, 2]
        dataset['Resv3_KF'] = new_seq[:, 3]
        dataset['Dist_KF'] = new_seq[:, 4]
        dataset['Velocity'] = dataset['Dist_KF'].diff() / 0.05  # calculate the velocity



    dataset.loc[dataset['Motor_KF'] < 0.05,'Motor_KF'] = 0
    dataset.loc[dataset['Brake_KF'] < 0.05,'Brake_KF'] = 0
    dataset.loc[dataset['Door_KF'] < 0,'Door_KF'] = 0
    dataset.loc[dataset['Resv3_KF'] < 0.05,'Resv3_KF'] = 0

    CarSeg_list = []
    DoorSeg_list = []
    ind_start = 0
    ind_end = 0
    ind_prestart = 0
    
    
    i = 1
    while i < (len(dataset)-1): # i < (len(dataset) - 120):
    
    
        if (dataset.loc[i, 'Brake'] == 0) & (dataset.loc[(i + 1):(i + 120), 'Brake'] > 0).all():
            #        seg_ind[i-10,0] = i # cycle starting index
            dataset.loc[i, 'seg_flag'] = 1  # use 1 to mark cycle starting point
            ind_start = i
    
            DoorSeg_list.append(dataset.iloc[ind_end:ind_start])  # Door motion cycle list
    
    
        elif (dataset.loc[(i - 120):(i - 1), 'Brake'] > 0).all() & (dataset.loc[i, 'Brake'] == 0):
    
            dataset.loc[i+1, 'seg_flag'] = 2  # use 2 to mark cycle ending point
            ind_end = i+1
    
            CarSeg_list.append(dataset.iloc[ind_start:ind_end])  # lift car traveling cycle list
    
            ind_prestart = ind_start
    
        i += 1
    
    if ind_start == ind_prestart:
        DoorSeg_list.append(dataset.iloc[ind_end:i])  # Door motion cycle list
        
        
    # 删除第一个不完整的car cycle
    if len(CarSeg_list)>0:
        if CarSeg_list[0].iloc[0]['Brake']>0:
    
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


def ACO(seq,end_time,RatedSpeed,event_list): # Ascending Car Overspeed

    if seq['Distance'].notnull().any() & ((seq['Distance'].iloc[-1] - seq['Distance'].iloc[0]) < -3):  # 判断电梯是否处于上升状态

        seq_overspeed = seq.loc[seq['Velocity'] > 1.15 * RatedSpeed] # When lift car is travelling upward and >115% of rated speed

        if len(seq_overspeed) > 40: # overspeed 超过2s
            log_text = {
                "time": str(end_time),
                "status ID": 2.2,
                "event": "ACO",
                 "description": "Lift ascending over speed ",
                    }
            event_list.append(log_text)
            
    return event_list

def UCM(seq,end_time,event_list):
    
    if seq['Distance'].notnull().any() & (np.abs(seq.iloc[-1]['Distance']-seq.iloc[0]['Distance']) > 1000.3): # 最近修改2021-12-20 car has travelled beyond door zone (+/- 300 mm of floor level)
        
        log_text = {
            "time": str(end_time),
            "status ID": 2.1,
            "event": "UCM",
            "description": "lift car has travelled beyond door zone (+/- 300 mm of floor level)",
                }
        event_list.append(log_text)      
    return event_list

def Brake_mag(seq,end_time,BrIsteady_Max,BrIsteady_Min,BrIpeak_Max,BrIpeak_Min,event_list): # Brake current magnitude
    BrakeI_steady = seq['Brake_KF'].iloc[36:-4].mean() #average brake steady current when car runs
    BrakeI_max = seq['Brake_KF'].max() # max brake current when car runs

    if (BrakeI_steady > BrIsteady_Max) | (BrakeI_steady < BrIsteady_Min): # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
              "description": "anomaly brake steady current magnitude:" + str(round(BrakeI_steady,2)),
                 }
         event_list.append(log_text)

    if (BrakeI_max > BrIpeak_Max) | (BrakeI_max < BrIpeak_Min): # out of the normal brake current range
         log_text = {
             "time": str(end_time),
             "status ID": 2.3,
             "event": "Brake Faults",
             "description": "anomaly brake steady current magnitude:" + str(round(BrakeI_max,2)),
                 }
         event_list.append(log_text)
         
    return event_list

def Motor_mag(seq,end_time,MotIpeak_Max,MotIpeak_Min,event_list): # Motor current magnitude
    MotorI_max = seq['Motor_KF'].iloc[25:75].mean()  # starting current of motor
    if (MotorI_max > MotIpeak_Max) | (MotorI_max < MotIpeak_Min): # out of the normal motor current range
         log_text = {
             "time": str(end_time),
             "status ID": 3.3,
             "event": "Motor anomaly",
             "description": "anomaly motor starting current magnitude:" + str(round(MotorI_max,2)),
                 }
         event_list.append(log_text)     
         
    return event_list

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

def do_action_LSTMAE(dataset, paras, CarSeg_list, DoorSeg_list):
    event_list = []
    
    BrIsteady_Max = paras['BrIsteady_Max']
    BrIsteady_Min = paras['BrIsteady_Min']
    BrIpeak_Max = paras['BrIpeak_Max']
    BrIpeak_Min = paras['BrIpeak_Min']  

    MotIpeak_Max = paras['MotIpeak_Max']
    MotIpeak_Min = paras['MotIpeak_Min']
    # thres_numDoor = paras['thres_numDoor']
    # line_Door = paras['line_Door']
    # MotZero_Rate = paras['MotZero_Rate']
    RatedSpeed = paras['RatedSpeed']    
    DoorError_max = paras['DoorError_max']
    MotorError_max = paras['MotorError_max']
    BrakeError_max = paras['BrakeError_max']
    

    ##计算loss
    criterion = torch.nn.MSELoss(reduction='sum')
    criterion2 = torch.nn.L1Loss(reduction='sum')
    ############################################################
    ################  判断 CarSeg_list ########################
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
            event_list = ACO(seq,end_time,RatedSpeed,event_list)
            event_list = Brake_mag(seq,end_time,BrIsteady_Max,BrIsteady_Min,BrIpeak_Max,BrIpeak_Min,event_list)
            event_list = Motor_mag(seq,end_time,MotIpeak_Max,MotIpeak_Min,event_list)

            ####################### AI #########################
            # seq.loc[seq['Motor'] < 0, 'Motor'] = 0
            # seq_motor = list(np.reshape(np.array(seq['Motor']), (1, -1)))
            # motor_list = motor_list + seq_motor
            seq.loc[seq['Motor'] < 0, 'Motor'] = 0
            event_list = LSTMFCN_Motor(seq,end_time,event_list)

            
            seq.loc[seq['Brake'] < 0, 'Brake'] = 0
            seq_brake = list(np.reshape(np.array(seq['Brake']), (1, -1)))
            brake_list = brake_list + seq_brake

            event_list = LSTMAE_Brake(criterion,brake_result,brake_list,CarSeg_list,BrakeError_max,event_list)





    ############################################################
    ################  判断 DoorSeg_list ########################
    ############################################################

### 文嘉用LSTM-AE 判断door anomaly ###################
    # door_list = []
    # door_result = []
    for i in range(len(DoorSeg_list)):
        seq = DoorSeg_list[i]
        
        end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
        
        ####################### Rules #########################

        event_list = UCM(seq,end_time,event_list)
        
        ####################### 2. MLSTM-FCN #########################
        
        event_list = LSTMFCN_Door(seq,end_time,event_list)

            ### Yujia ###################
    return event_list


def do_action_RULES(dataset, paras, CarSeg_list, DoorSeg_list):
    event_list = []
    
    BrIsteady_Max = paras['BrIsteady_Max']
    BrIsteady_Min = paras['BrIsteady_Min']
    BrIpeak_Max = paras['BrIpeak_Max']
    BrIpeak_Min = paras['BrIpeak_Min']  
    MotIpeak_Max = paras['MotIpeak_Max']
    MotIpeak_Min = paras['MotIpeak_Min']
    thres_numDoor = paras['thres_numDoor']
    line_Door = paras['line_Door']
    MotZero_Rate = paras['MotZero_Rate']
    RatedSpeed = paras['RatedSpeed']
    
 

    ################  判断 CarSeg_list ########################
    if len(CarSeg_list) > 0:
        for i in range(len(CarSeg_list)):
            seq = CarSeg_list[i]
            end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')
            
            event_list = ACO(seq,end_time,RatedSpeed,event_list)
            event_list = Brake_mag(seq,end_time,BrIsteady_Max,BrIsteady_Min,BrIpeak_Max,BrIpeak_Min,event_list)
            event_list = Motor_mag(seq,end_time,MotIpeak_Max,MotIpeak_Min,event_list)

            # if len(seq.loc[seq['Motor']==0])/len(seq) > MotZero_Rate: # 电机电流=0的比例大于一定比例, 判定为电机异常
            #     log_text = {
            #         "time": str(end_time),
            #         "status ID": 3.3,
            #         "event": "Motor anomaly",
            #         "description": "anomaly motor pattern",
            #             }
            #     event_list.append(log_text)


    ################  判断 DoorSeg_list ########################
    for i in range(len(DoorSeg_list)):    
        seq = DoorSeg_list[i]
        end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S.%f')


        event_list = UCM(seq,end_time,event_list)
        event_list = Door_nos(seq,end_time,thres_numDoor,line_Door,event_list)


    return event_list


            
def do_action_SafetyCircuit(dataset,paras):
    
    event_list = []
    seq_lock = dataset.loc[(dataset['Safety']>=paras['Locked_SafetyI_Min']) & (dataset['Safety']<=paras['Locked_SafetyI_Max'])]
    seq_trip = dataset.loc[(dataset['Safety']>=paras['Tripped_SafetyI_Min']) & (dataset['Safety']<=paras['Tripped_SafetyI_Max'])]
    seq_dip = dataset.loc[(dataset['Safety']>=paras['Voltage_Dip_Min']) & (dataset['Safety']<=paras['Voltage_Dip_Max'])]
    seq_inspection_run = dataset.loc[(dataset['Safety']>=paras['InspectionRun_Min']) & (dataset['Safety']<=paras['InspectionRun_Max'])]
    seq_inspection_stop = dataset.loc[(dataset['Safety']>=paras['InspectionStop_Min']) & (dataset['Safety']<=paras['InspectionStop_Max'])]
    


    if len(seq_dip)>=20:        
        log_text = {
            "time": str(seq_dip.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 2.5,
            "event": " Voltage dip",
            "description": "Urgent - Voltage dip",
                }
        event_list.append(log_text) # 把电压骤降事件存到event_list里 


    elif len(seq_trip)>=20:        
        log_text = {
            "time": str(seq_trip.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 2.6,
            "event": "Safety tripped (Idle)",
            "description": "Urgent - Safety tripped when lift is in idle)",
                }
        event_list.append(log_text) # 把trip事件存到event_list里   
        
    elif len(seq_lock)>=20:
        log_text = {
            "time": str(seq_lock.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 5,
            "event": "Locked",
            "description": "Lock mode - out of service",
                            }
        event_list.append(log_text) # 把锁机事件存到event_list里   
        
    elif len(seq_inspection_run)>=40:
        log_text = {
            "time": str(seq_inspection_run.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 4,
            "event": "Inspection mode",
            "description": "Inspection mode - maintenance",
                            }
        event_list.append(log_text) # 把检修事件存到event_list里   
        
    elif len(seq_inspection_stop)>=40:
        log_text = {
            "time": str(seq_inspection_stop.index[1].strftime('%Y-%m-%d %H:%M:%S.%f')),
            "status ID": 4,
            "event": "Inspection mode",
            "description": "Inspection mode - maintenance",
                            }
        event_list.append(log_text) # 把检修事件存到event_list里   
        
        
    return event_list
