#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:10:51 2022

@author: chaisongjian
"""

import numpy as np

# 定义全局参数

MissingData_Rate = 0.5
thres_numDoor = 6
# 定义各个电梯的关键参数

############################################################
######################  EMSD ############################### 
############################################################
paras_EMSDHQL12 = {
    "RMU_ID": 21,
    "Lift_Name": 'EMSDHQL12',
    "Lift_ID": 10,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.06, 0.11]),
    "Tripped_SafetyI_range": np.array([0, 0.06]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.21, 0.27]),
    "Idle_SafetyI_range":np.array([0.16, 0.20]),
    "Dooropen_SafetyI_range":np.array([0.12, 0.16]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.2, 2]),
    "BrIpeak_Range": np.array([0.2, 3]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.29,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.1, 0.75]),
    "RatedSpeed": 1.6,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F'], 
    "Floor_height": [43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }

paras_EMSDHQL06 = {
    "RMU_ID": 19,
    "Lift_Name": 'EMSDHQL06',
    "Lift_ID": 530990078732,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.06, 0.11]),
    "Tripped_SafetyI_range": np.array([0.0, 0.06]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.23, 0.29]),
    "Idle_SafetyI_range":np.array([0.17, 0.21]),
    "Dooropen_SafetyI_range":np.array([0.12, 0.17]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.2, 2]),
    "BrIpeak_Range": np.array([0.2, 3]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.3,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,    
    "DrIpeak_Range": np.array([0.1, 0.75]),
    "RatedSpeed": 1.6,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 25,
    # "Floor": np.array([np.nan,41.16,34.89,27.73,21.87,14.83,8.68,3.18,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,39.48,33.43,26.31,20.64,13.58,6.75,1.87,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F'], 
    "Floor_height": [39.48,33.43,26.31,20.64,13.58,6.75,0.54],# 20240110 更新 ，‘6/F’变为0.54，因为lidar问题。
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }



paras_EMSDHQL08 = {
    "RMU_ID": 28,
    "Lift_Name": 'EMSDHQL08',
    "Lift_ID": 11,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.03, 0.11]),
    "Tripped_SafetyI_range": np.array([0.0, 0.03]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.21, 0.30]),
    "Idle_SafetyI_range":np.array([0.17, 0.21]),
    "Dooropen_SafetyI_range":np.array([0.11, 0.17]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.2, 2]),
    "BrIpeak_Range": np.array([0.2, 3]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.23,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.1, 0.75]),
    "RatedSpeed": 1.6,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 25,
    # "Floor": np.array([np.nan,45.71,40.00,32.41,27.38,20.47,14.32,8.62,2.26,np.nan,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,44.77,39.03,35.71,31.93,26.19,19.26,13.16,7.49,1.25,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','M/F','2/F','3/F','4/F','5/F','6/F','7/F'], 
    "Floor_height": [44.77,39.03,35.71,31.93,26.19,19.26,13.16,7.49,1.25],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 1,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-05-28
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

# 44.77,39.03,35.71,31.93,26.19,19.26,13.22,7.49,1.25

paras_EMSDHQL10 = {
    "RMU_ID": 20,
    "Lift_Name": 'EMSDHQL10',
    "Lift_ID": 521661388479,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.05, 0.11]),
    "Tripped_SafetyI_range": np.array([0.0, 0.05]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.20, 0.24]),
    "Idle_SafetyI_range":np.array([0.16, 0.19]),
    "Dooropen_SafetyI_range":np.array([0.12, 0.16]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.2, 2]),
    "BrIpeak_Range": np.array([0.2, 3]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.23,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.1, 0.75]),
    "RatedSpeed": 1.6,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 25,
    # "Floor": np.array([np.nan,45.33,39.53,32.50,26.79,19.83,13.58,7.29,1.07,np.nan,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F'], 
    "Floor_height": [45.33,39.53,32.50,26.79,19.83,13.58,7.29,1.07],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_EMSDHQL09 = {
    "RMU_ID": 10012,
    "Lift_Name": 'EMSDHQL09',
    "Lift_ID": 25,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.07, 0.10]),
    "Tripped_SafetyI_range": np.array([0.10, 0.14]),
    "Run_SafetyI_range":np.array([0.27, 0.30]),
    "Idle_SafetyI_range":np.array([0.20, 0.24]),
    "Dooropen_SafetyI_range":np.array([0.15, 0.19]),
    "Voltage_Dip_range": np.array([0.0, 0.07]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.2, 2]),
    "BrIpeak_Range": np.array([0.2, 3]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.3,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.1, 0.75]),
    "RatedSpeed": 1.6,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 25,
    # "Floor": np.array([np.nan,45.94,40.07,32.60,26.64,19.75,13.83,7.36,4.23,np.nan,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F'], 
    "Floor_height": [45.94,40.07,32.60,26.64,19.75,13.83,7.36,4.23],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # # 2023-06-16
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


paras_EMSDHQL11 = {
    "RMU_ID": 2,
    "Lift_Name": 'EMSDHQL11',
    "Lift_ID": 655148485456,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.09, 0.11]),
    "Tripped_SafetyI_range": np.array([0.06, 0.09]),
    "Voltage_Dip_range": np.array([0, 0.06]),
    "Run_SafetyI_range":np.array([0.22, 0.25]),
    "Idle_SafetyI_range":np.array([0.16, 0.20]),
    "Dooropen_SafetyI_range":np.array([0.12, 0.15]),
    "InspectionRun_range": np.array([-4, -2]),
    "InspectionStop_range": np.array([-4, -2]),
    "BrIsteady_Range": np.array([0.8, 2.0]),
    "BrIpeak_Range": np.array([1.2, 2.5]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.45,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.4, 1.1]),
    "RatedSpeed": 1.6,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,63.29,57.48,50.30,44.48,37.61,31.49,25.72,19.86,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F'], 
    "Floor_height": [63.29,57.48,50.30,44.48,37.61,31.49,25.72,19.86],
    "Position_sensor": 2, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.005,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22  
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


paras_EMSDHQL5 = {
    "RMU_ID": 3,
    "Lift_Name": 'EMSDHQL5',
    "Lift_ID": 655148485457,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.07, 0.10]),
    "Tripped_SafetyI_range": np.array([0.05, 0.07]),
    "Voltage_Dip_range": np.array([0, 0.05]),
    "Run_SafetyI_range":np.array([0.23, 0.30]),
    "Idle_SafetyI_range":np.array([0.17, 0.21]),
    "Dooropen_SafetyI_range":np.array([0.13, 0.16]),
    "InspectionRun_range": np.array([-4, -2]),
    "InspectionStop_range": np.array([-4, -2]),
    "BrIsteady_Range": np.array([0.8, 2.0]),
    "BrIpeak_Range": np.array([1.2, 2.5]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.45,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.4, 1.1]),
    "RatedSpeed": 1.6,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,63.26,57.47,50.31,44.52,37.62,31.47,25.71,19.86,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F'], 
    "Floor_height": [63.26,57.47,50.31,44.52,37.62,31.47,25.71,19.86],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.005, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

############################################################
######################   PV  ############################### 
############################################################

paras_PVTower1L1 = {
    "RMU_ID": 31,
    "Lift_Name": 'PVTower1L1',
    "Lift_ID": 20,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([-2, -1]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([0.01, 0.035]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.75, 2]),
    "BrIpeak_Range": np.array([1.0, 4.2]),
    "MotIpeak_Range": np.array([0, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.15,
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 2.0,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,29.3,22.01,19.17,16.31,13.46,10.6,7.77,4.87,1.94]),# LG,G,1,2,3,4,5,6,7,8,
    # "Floor": np.nan,
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F','8/F'], 
    "Floor_height": [29.3,22.01,19.17,16.31,13.46,10.6,7.77,4.87,1.94],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_PVTower1L2 = {
    "RMU_ID": 33,
    "Lift_Name": 'PVTower1L2',
    "Lift_ID": 21,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([-2, -1]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([0.01, 0.035]), 
    "Voltage_Dip_range": np.array([-2, -1]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.75, 2]),
    "BrIpeak_Range": np.array([1.0, 4.2]),
    "MotIpeak_Range": np.array([0, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.2,
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 2.0,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,29.08,21.94,19.16,16.35,13.52,10.64,7.73,4.85,1.97]),# LG,G,1,2,3,4,5,6,7,8
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F','8/F'], 
    "Floor_height": [29.08,21.94,19.16,16.35,13.52,10.64,7.73,4.85,1.97],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_PVTower2L1 = {
    "RMU_ID": 34,
    "Lift_Name": 'PVTower2L1',
    "Lift_ID": 23,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.01, 0.1]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([-1, 0.02]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.9, 1.4]),
    "BrIpeak_Range": np.array([2.0, 2.5]),
    "MotIpeak_Range": np.array([10, 60]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.32,
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([0.3,1.0]),
    "RatedSpeed": 2.0,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,41.49,35.82,30.66,27.81,24.97,22.17,19.27,16.43,13.58,10.78,7.97,5.12,2.32]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F','8/F','9/F','10/F','11/F','12/F'], 
    "Floor_height": [41.49,35.82,30.66,27.81,24.97,22.17,19.27,16.43,13.58,10.78,7.97,5.12,2.32],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan, #0.009,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_PVTower2L2 = {
    "RMU_ID": 36,
    "Lift_Name": 'PVTower2L2 ',
    "Lift_ID": 22,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([-2, -1]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([-1, 0.02]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.9, 1.6]),
    "BrIpeak_Range": np.array([2.0, 2.6]),
    "MotIpeak_Range": np.array([10, 60]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.29,
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 2.0,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,42.04,36.28,31.08,28.38,25.52,22.72,19.89,17.04,14.18,11.33,8.47,5.67,2.8]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F','8/F','9/F','10/F','11/F','12/F'], 
    "Floor_height": [42.04,36.28,31.08,28.38,25.52,22.72,19.89,17.04,14.18,11.33,8.47,5.67,2.8],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
     "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
   }


############################################################
######################  WCH  ############################### 
############################################################

paras_WCHEWL1 = {
    "RMU_ID": 17,
    "Lift_Name": 'WCHEWL1',
    "Lift_ID": 14,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.1, 0.16]),
    "Tripped_SafetyI_range": np.array([0.16, 0.2]),
    "Voltage_Dip_range": np.array([0, 0.1]),
    "Run_SafetyI_range":np.array([50, 51]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([-2, -1]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([1.0, 1.8]),
    "BrIpeak_Range": np.array([1.0, 3.0]),
    "MotIpeak_Range": np.array([20, 60]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.07,
    "DoorOpen_ref": 2, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.08, 0.35]),
    "RatedSpeed": 1.5,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,36.33,31.16,26.16,20.66,16.14,11.64,7.16,2.09,np.nan,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F'], 
    "Floor_height": [36.33,31.16,26.16,20.66,16.14,11.64,7.16,2.09],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":0.005,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_WCHEWL2 = {
    "RMU_ID": 18,
    "Lift_Name": 'WCHEWL2',
    "Lift_ID": 15,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.1, 0.16]),
    "Tripped_SafetyI_range": np.array([0.16, 0.2]),
    "Voltage_Dip_range": np.array([0, 0.1]),
    "Run_SafetyI_range":np.array([50, 51]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([-2, -1]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.5, 1.8]),
    "BrIpeak_Range": np.array([1.0, 3.0]),
    "MotIpeak_Range": np.array([20, 60]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.07,
    "DoorOpen_ref": 2, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.08, 0.35]),
    "RatedSpeed": 1.5,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,36.33,31.16,26.16,20.66,16.14,11.64,7.16,2.09,np.nan,np.nan,np.nan,np.nan,np.nan]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.nan,
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F'], # Dist数据失真，LiDAR 反光板可能需要擦拭清理 
    "Floor_height": [36.33,31.16,26.16,20.66,16.14,11.64,7.16,2.09], # Dist数据失真，LiDAR 反光板可能需要擦拭清理 
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":0.005,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


paras_WCHMWL13 = {
    "RMU_ID": 22,
    "Lift_Name": 'WCHMWL13',
    "Lift_ID": 19,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2,-1]),
    "Tripped_SafetyI_range": np.array([0, 0.01]),
    "Voltage_Dip_range": np.array([-2,-1]),
    "Run_SafetyI_range":np.array([0.28, 0.36]),
    "Idle_SafetyI_range":np.array([0.08, 0.14]),
    "Dooropen_SafetyI_range":np.array([0.02, 0.08]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.4, 1.2]),
    "BrIpeak_Range": np.array([0.6, 1.0]),
    "MotIpeak_Range": np.array([0, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": np.nan,
    "line_Door": np.nan,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": np.nan,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 2.5,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,np.nan,np.nan,np.nan,np.nan,57.38,np.nan,
    #                np.nan,np.nan,np.nan,33.22,27.81,23.87,19.83,15.76,
    #                11.79,7.80,3.77]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    "Floor_level": ['3/F','8/F','9/F','10/F','11/F','12/F','13/F','14/F','15/F'], 
    "Floor_height": [57.38,33.22,27.81,23.87,19.83,15.76,11.79,7.80,3.77],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


paras_WCHMWL15 = {
    "RMU_ID": 23,
    "Lift_Name": 'WCHMWL15',
    "Lift_ID": 18,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2,-1]),
    "Tripped_SafetyI_range": np.array([0, 0.01]),
    "Voltage_Dip_range": np.array([-2,-1]),
    "Run_SafetyI_range":np.array([0.22, 0.28]),
    "Idle_SafetyI_range":np.array([0.07, 0.14]),
    "Dooropen_SafetyI_range":np.array([0.02, 0.07]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.6, 1.2]),
    "BrIpeak_Range": np.array([1.5, 2.2]),
    "MotIpeak_Range": np.array([0, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": np.nan,
    "line_Door": np.nan,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": np.nan,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 2.5,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,np.nan,np.nan,np.nan,np.nan,58.28,np.nan,
    #                    np.nan,np.nan,np.nan,34.13,28.75,24.66,19.98,15.93,
    #                    12.47,8.22,3.86]),# LG,G,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    "Floor_level": ['3/F','8/F','9/F','10/F','11/F','12/F','13/F','14/F','15/F'], 
    "Floor_height": [58.28,34.13,28.75,24.66,19.98,15.93,12.47,8.22,3.86],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }





paras_WCHWWL8 = {
    "RMU_ID": 29,
    "Lift_Name": 'WCHWWL8',
    "Lift_ID": 16,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0, 0.12]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.21, 0.26]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([0.16, 0.21]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.9, 1.3]),
    "BrIpeak_Range": np.array([0.9, 1.4]),
    "MotIpeak_Range": np.array([0, 60]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.09,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 3.0,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.nan,
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 0, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_WCHWWL9 = {
    "RMU_ID": 30,
    "Lift_Name": 'WCHWWL9',
    "Lift_ID": 17,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0, 0.12]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.21, 0.26]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([0.16, 0.21]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.7, 1.1]),
    "BrIpeak_Range": np.array([1.7, 2.5]),
    "MotIpeak_Range": np.array([0, 60]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.18,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 3.0,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.nan,
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 0, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


############################################################
######################   WPS  ############################## 
############################################################
paras_WPS1 = {
    "RMU_ID": 24,
    "Lift_Name": 'WPS1',
    "Lift_ID": 24,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0.20, 0.22]),
    "Voltage_Dip_range": np.array([0, 0.15]),
    "Run_SafetyI_range":np.array([0.76, 0.82]),
    "Idle_SafetyI_range":np.array([0.46, 0.52]),
    "Dooropen_SafetyI_range":np.array([0.28, 0.34]), 
    "InspectionRun_range": np.array([0.65, 0.67]),
    "InspectionStop_range": np.array([0.33, 0.34]),
    "BrIsteady_Range": np.array([0.35, 2.5]),
    "BrIpeak_Range": np.array([0.35, 2.5]),
    "MotIpeak_Range": np.array([0, 60]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":12,
    "line_Door": 0.2,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([np.nan, np.nan]),
    "RatedSpeed": 1.6,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,21.38,14.34,11.57,8.51,5.31,1.3,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,17.69,14.16,10.89,7.67,4.55,2.12,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F'], 
    "Floor_height": [17.69,14.16,10.89,7.67,4.55,2.12],
    "Position_sensor": 0, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


############################################################
######################   FSD  ############################### 
############################################################

paras_FSDL1 = {
    "RMU_ID": 10,
    "Lift_Name": 'FSDL1',
    "Lift_ID": 655148495843,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-1, 0.13]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.30, 0.33]),
    "Idle_SafetyI_range":np.array([0.20, 0.24]),
    "Dooropen_SafetyI_range":np.array([0.16, 0.19]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([0.085, 0.111]),
    "BrIsteady_Range": np.array([1.4, 2.5]),
    "BrIpeak_Range": np.array([1.4, 2.5]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.6,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.6, 1.35]),
    "RatedSpeed": 1.75,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,40.05,36.01,31.94,27.92,24.52,21.13,17.74,14.35,10.95,7.56,4.16,0.81,np.nan]), # G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.nan,
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F','8/F','9/F','10/F','11/F'], 
    "Floor_height": [40.05,36.01,31.94,27.92,24.52,21.13,17.74,14.35,10.95,7.56,4.16,0.81],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.003, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_FSDL2 = {
    "RMU_ID": 11,
    "Lift_Name": 'FSDL2',
    "Lift_ID": 655148495844,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-1, 0.10]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.22, 0.26]),
    "Idle_SafetyI_range":np.array([0.15, 0.18]),
    "Dooropen_SafetyI_range":np.array([0.11, 0.14]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([0.09, 0.094]),
    "BrIsteady_Range": np.array([1.6, 2.3]),
    "BrIpeak_Range": np.array([1.6, 2.3]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.835,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([0.6, 1.35]),
    "RatedSpeed": 1.75,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,40.03,36.02,31.97,27.90,24.52,21.12,17.74,14.32,10.92,7.53,4.15,0.78,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F','8/F','9/F','10/F','11/F'], 
    "Floor_height": [40.03,36.02,31.97,27.90,24.52,21.12,17.74,14.32,10.92,7.53,4.15,0.78],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.003, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

paras_FSDL3 = {
    "RMU_ID": 27,
    "Lift_Name": 'FSDL3',
    "Lift_ID": 655148495841,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-1, 0.135]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.28, 0.33]),
    "Idle_SafetyI_range":np.array([0.21, 0.24]),
    "Dooropen_SafetyI_range":np.array([0.16, 0.2]), 
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-3, -4]),
    "BrIsteady_Range": np.array([1.6, 2.5]),
    "BrIpeak_Range": np.array([1.6, 2.5]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.837,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([0.6, 1.35]),
    "RatedSpeed": 1.75,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,40.05,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,14.32,10.92,7.52,4.14,0.75,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','7/F','8/F','9/F','10/F','11/F'], 
    "Floor_height": [40.05,14.32,10.92,7.52,4.14,0.75],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.001, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }



paras_FSDL4 = {
    "RMU_ID": 12,
    "Lift_Name": 'FSDL4',
    "Lift_ID": 655148495845,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-1, 0.12]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.31, 0.35]),
    "Idle_SafetyI_range":np.array([0.17, 0.21]),
    "Dooropen_SafetyI_range":np.array([0.13, 0.16]), 
    "InspectionRun_range": np.array([0.21, 0.26]),
    "InspectionStop_range": np.array([0.07, 0.11]),
    "BrIsteady_Range": np.array([1.5, 2.5]),
    "BrIpeak_Range": np.array([1.5, 2.5]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.85,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.6, 1.6]),
    "RatedSpeed": 1.75,        
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,40.11,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,14.37,10.97,7.55,4.18,0.79,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','7/F','8/F','9/F','10/F','11/F'], 
    "Floor_height": [40.11,14.37,10.97,7.55,4.18,0.79],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.005, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }





############################################################
###   Old Bailet Street Police Married Quarters ############
############################################################
paras_OTISL3 = {
    "RMU_ID": 16,
    "Lift_Name": 'OTISL3',
    "Lift_ID": 1,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.12, 0.23]),
    "Tripped_SafetyI_range": np.array([0.10, 0.12]),
    "Voltage_Dip_range": np.array([0, 0.085]),
    "Run_SafetyI_range":np.array([0.40, 0.48]),
    "Idle_SafetyI_range":np.array([0.29, 0.32]),
    "Dooropen_SafetyI_range":np.array([0.26, 0.29]),
    "InspectionRun_range": np.array([0.35, 0.37]),
    "InspectionStop_range": np.array([0.085, 0.10]),
    "BrIsteady_Range": np.array([0.7, 2.3]),
    "BrIpeak_Range": np.array([0.7, 2.3]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.91,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,  
    "DrIpeak_Range": np.array([0.6, 1.4]),
    "RatedSpeed": 1.60,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.nan,
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 0, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": np.nan, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }

############################################################
################   Tai Po Government Building ##############
############################################################
paras_TPGL1 = {
    "RMU_ID": 13,
    "Lift_Name": 'TPGL1',
    "Lift_ID": 655148495846,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0.0, 0.005]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.30, 0.36]),
    "Idle_SafetyI_range":np.array([0.02, 0.03]),
    "Dooropen_SafetyI_range":np.array([0, 0.0199]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-2, -1]),
    "BrIsteady_Range": np.array([1.7, 2.7]),
    "BrIpeak_Range": np.array([1.7, 2.7]),
    "MotIpeak_Range": np.array([10, 70]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.21,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([0.1, 0.4]),
    "RatedSpeed": 1.60,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.nan,
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 0, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": np.nan, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


############################################################
#########   Queen  Elizabeth Hospital #######################
############################################################
paras_QEBL8 = {
    "RMU_ID": 15,
    "Lift_Name": 'QEBL8',
    "Lift_ID": 655148495848,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-5, -4]),
    "Tripped_SafetyI_range": np.array([-1, 0.0001]), 
    "Voltage_Dip_range": np.array([-5, -4]),
    "Run_SafetyI_range":np.array([0.07, 0.09]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([-3, -2]),
    "InspectionRun_range": np.array([-4, -3]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([1.8, 3.5]),
    "BrIpeak_Range": np.array([3.0, 5.0]),
    "RunField_Range": np.array([14, 25]), # run field current range 
    "FullField_Range": np.array([21, 26]), # full field current range 
    "ArmaturePeak_Range": np.array([30, 300]), # armature peak current range 
    "ArmatureStart_Range": np.array([-300, 300]), # armature start current range 
    "ArmatureBrake_Range": np.array([-300, 300]), # armature brake current range 
    "ArmatureSteady_Range": np.array([-300, 300]), # armature steady current range 
    "thres_numDoor":thres_numDoor,
    "line_Door": 1.05,
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 2, 
    "DrIpeak_Range": np.array([1.0, 2.1]),
    "RatedSpeed": 2.50,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([43.9,40.23,36.3,32.95,29.62,26.25,22.87,19.54,16.17,12.84,9.47,6.1,2.77,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['LG/F','G/F','1/F','2/F','3/F','4/F','5/F','6/F','7/F','8/F','9/F','10/F','11/F'], 
    "Floor_height": [43.9,40.23,36.3,32.95,29.62,26.25,22.87,19.54,16.17,12.84,9.47,6.1,2.77],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.015, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-02-22
    "Motor_type": "DC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }



############################################################
################   Queen  Mary Hospital #######################
############################################################
paras_QML3 = {
    "RMU_ID": 14,
    "Lift_Name": 'QML3',
    "Lift_ID": 655148495847,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0.0, 0.01]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.14, 0.16]), # 2023-03-04
    "Idle_SafetyI_range":np.array([0.12, 0.14]), # 2022-12-19
    "Dooropen_SafetyI_range":np.array([0.04, 0.08]), # 2023-03-04
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-2, -1]),
    "BrIsteady_Range": np.array([1.1, 1.65]),
    "BrIpeak_Range": np.array([1.1, 1.9]),
    "MotIpeak_Range": np.array([2, 85]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.75, 
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60, #2023-9-19
    "DoorWaveform_type": 1,
    "DrIpeak_Range": np.array([0.1, 2.3]),
    "RatedSpeed": 1.60,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,27.07,22.82,18.55,14.31,10.01,5.76,1.47,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": ['G/F','1/F','2/F','3/F','4/F','5/F','6/F'], 
    "Floor_height": [27.07,22.82,18.55,14.31,10.01,5.76,1.47],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.02, # 2023-03-04
    "AI_FLAG": 0, # 2023-02-22
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


############################################################
################### Hing Wah Estate #######################
############################################################


paras_HWEL1 = {
    "RMU_ID": 26,
    "Lift_Name": 'HWE L1',
    "Lift_ID": 541434160882,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0.0, 0.01,-1,0.01]), # 后两个为door current 在trip的时候的范围
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.07, 0.10]),
    "Idle_SafetyI_range":np.array([0.025, 0.035]),
    "Dooropen_SafetyI_range":np.array([0.015, 0.025]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-2, -1]),
    "BrIsteady_Range": np.array([0.45, 0.9]),
    "BrIpeak_Range": np.array([0.45, 0.9]),
    "MotIpeak_Range": np.array([2, 65]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.9,
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 4,
    "DrIpeak_Range": np.array([0.4, 1.6]),
    "RatedSpeed": 1.50,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,np.nan,np.nan,np.nan,np.nan,62.28,np.nan,np.nan,54.58,np.nan,
    #                    49.43,np.nan,43.87,np.nan,38.77,np.nan,33.66,np.nan,28.58,
    #                    np.nan,23.41,np.nan,18.28,np.nan,13.15,np.nan,np.nan,5.39,2.79]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
    "Floor_level": ['4/F','7/F','9/F','11/F','13/F','15/F','17/F','19/F','21/F','23/F','26/F','27/F'], 
    "Floor_height": [62.28,54.58,49.43,43.87,38.77,33.66,28.58,23.41,18.28,13.15,5.39,2.79],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.03, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-05-28,'HWE100%_model.h5'needs umprovement
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }


paras_HWEL2 = {
    "RMU_ID": 32,
    "Lift_Name": 'HWE L2',
    "Lift_ID": 541434166958,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0.0, 0.025,-1,0.01]), # 后两个为door current 在trip的时候的范围
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.06, 0.09]),
    "Idle_SafetyI_range":np.array([0.025, 0.035]),
    "Dooropen_SafetyI_range":np.array([0.015, 0.025]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-2, -1]),
    "BrIsteady_Range": np.array([0.45, 0.9]),
    "BrIpeak_Range": np.array([0.45, 0.9]),
    "MotIpeak_Range": np.array([2, 65]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.8,
    "DoorOpen_ref": 0, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 70,
    "DoorWaveform_type": 4,
    "DrIpeak_Range": np.array([0.4, 1.5]),
    "RatedSpeed": 1.50,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,np.nan,np.nan,np.nan,np.nan,62.5,np.nan,57.28,np.nan,52.18,
    #                    49.62,46.62,np.nan,41.42,np.nan,36.27,np.nan,31.16,np.nan,np.nan,
    #                    np.nan,20.99,np.nan,15.76,np.nan,10.59,np.nan,np.nan,2.85]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
    "Floor_level": ['4/F','6/F','8/F','9/F','10/F','12/F','14/F','16/F','20/F','22/F','24/F','27/F'], 
    "Floor_height": [62.5,57.28,52.18,49.62,46.62,41.42,36.27,31.16,20.99,15.76,10.59,2.85],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": 0.002, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-05-28,'HWE100%_model.h5'needs umprovement
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 2 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s        
    }



############################################################
######################### VTC ##############################
############################################################

paras_VTCL4 = {
    "RMU_ID": 49,
    "Lift_Name": 'VTC L4',
    "Lift_ID": 561141816017,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0.0, 0.01]), 
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.23, 0.30]),
    "Idle_SafetyI_range":np.array([0.08, 0.15]),
    "Dooropen_SafetyI_range":np.array([0.01, 0.05]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-2, -1]),
    "BrIsteady_Range": np.array([1.8, 3.4]),
    "BrIpeak_Range": np.array([4.2, 6.0]),
    "MotIpeak_Range": np.array([2, 65]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 2.0,
    "thres_DoorOpenDuration": 30,
    "DoorWaveform_type": 0, # door OP 
    "DoorOpenLong_FLAG": 0, # 不判断
    "DrIpeak_Range": np.array([2, 3.5]),
    "RatedSpeed": 1.50,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,np.nan,np.nan,np.nan,np.nan,23.16,18.64,14.18,9.64,5.18,
    #                    0.66,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
    #                    np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
    "Floor_level": ['4/F','5/F','6/F','7/F','8/F','9/F'], 
    "Floor_height": [23.16,18.64,14.18,9.64,5.18,0.66],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": np.nan, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-05-28,'HWE100%_model.h5'needs umprovement
    "Motor_type": "AC"
    }




paras_VTCL5 = {
    "RMU_ID": 50,
    "Lift_Name": 'VTC L5',
    "Lift_ID": 541434195678,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([0.0, 0.03]), 
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.6, 0.8]),
    "Idle_SafetyI_range":np.array([0.16, 0.24]),
    "Dooropen_SafetyI_range":np.array([0.05, 0.16]),
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-2, -1]),
    "BrIsteady_Range": np.array([0.5, 1.4]),
    "BrIpeak_Range": np.array([0.5, 1.4]),
    "MotIpeak_Range": np.array([2, 80]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor":thres_numDoor,
    "line_Door": 0.25,
    "thres_DoorOpenDuration": 30,
    "DoorWaveform_type": 2,
    "DoorOpenLong_FLAG": 0, # 不判断
    "DrIpeak_Range": np.array([0.3, 0.6]),
    "RatedSpeed": 1.50,          
    "DoorError_max": np.nan,
    "MotorError_max": np.nan,
    "BrakeError_max": np.nan,
    # "Floor": np.array([np.nan,np.nan,np.nan,np.nan,np.nan,31.54,25.55,19.56,13.56,7.63,
    #                    1.64,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
    #                    np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
    "Floor_level": ['4/F','5/F','6/F','7/F','8/F','9/F'], 
    "Floor_height": [31.54,25.55,19.56,13.56,7.63,1.64],
    "Position_sensor": 1, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3": np.nan, 
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-05-28,'HWE100%_model.h5'needs umprovement
    "Motor_type": "AC"
    }



############################################################
############# Chai Wan Police Married Quarters  ############
############################################################



paras_CWPQL1 = {
    "RMU_ID": 35,
    "Lift_Name": 'L1',
    "Lift_ID": 543940604489,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.1, 0.15]),
    "Idle_SafetyI_range":np.array([0.01, 0.05]),
    "Dooropen_SafetyI_range":np.array([-0.01, 0.01]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.8, 1.5]),
    "BrIpeak_Range": np.array([1.2, 1.8]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.68,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.7]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }




paras_CWPQL2 = {
    "RMU_ID": 37,
    "Lift_Name": 'L2',
    "Lift_ID": 543940636830,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.1, 0.15]),
    "Idle_SafetyI_range":np.array([0.03, 0.07]),
    "Dooropen_SafetyI_range":np.array([-0.01, 0.01]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([1.0, 1.8]),
    "BrIpeak_Range": np.array([1.4, 2.4]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.57,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.7]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }





paras_CWPQL3 = {
    "RMU_ID": 38,
    "Lift_Name": 'L3',
    "Lift_ID": 543940644429,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.1, 0.15]),
    "Idle_SafetyI_range":np.array([0.02, 0.07]),
    "Dooropen_SafetyI_range":np.array([-0.01, 0.01]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([1.0, 1.8]),
    "BrIpeak_Range": np.array([1.4, 2.2]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.6,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.5]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }


paras_CWPQL4 = {
    "RMU_ID": 39,
    "Lift_Name": 'L4',
    "Lift_ID": 543940653477,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.07, 0.13]),
    "Idle_SafetyI_range":np.array([0.045, 0.06]),
    "Dooropen_SafetyI_range":np.array([0.02, 0.045]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.8, 1.3]),
    "BrIpeak_Range": np.array([1.0, 1.6]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.5,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.1]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }



paras_CWPQL5 = {
    "RMU_ID": 40,
    "Lift_Name": 'L5',
    "Lift_ID": 543940661154,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.1, 0.15]),
    "Idle_SafetyI_range":np.array([0.01, 0.05]),
    "Dooropen_SafetyI_range":np.array([-0.01, 0.01]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.7, 1.2]),
    "BrIpeak_Range": np.array([1.1, 1.7]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.5,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.2]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }





paras_CWPQL6 = {
    "RMU_ID": 41,
    "Lift_Name": 'L6',
    "Lift_ID": 561147655834,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.1, 0.15]),
    "Idle_SafetyI_range":np.array([0.02, 0.05]),
    "Dooropen_SafetyI_range":np.array([-0.01, 0.02]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([1.2, 1.8]),
    "BrIpeak_Range": np.array([1.7, 2.6]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.5,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.3]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }




paras_CWPQL7 = {
    "RMU_ID": 42,
    "Lift_Name": 'L7',
    "Lift_ID": 543940678756,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.07, 0.12]),
    "Idle_SafetyI_range":np.array([0.03, 0.05]),
    "Dooropen_SafetyI_range":np.array([0, 0.03]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0.8, 1.3]),
    "BrIpeak_Range": np.array([1, 1.7]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.42,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.3]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }


paras_CWPQL8 = {
    "RMU_ID": 43,
    "Lift_Name": 'L8',
    "Lift_ID": 543940688064,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([0.07, 0.13]),
    "Idle_SafetyI_range":np.array([0.03, 0.06]),
    "Dooropen_SafetyI_range":np.array([-0.01, 0.01]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([1, 1.6]),
    "BrIpeak_Range": np.array([1.5, 2.3]),
    "MotIpeak_Range": np.array([2, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.52,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0.0, 1.4]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }


############################################################
########################### Guangzhou  ##########################
############################################################
# 测试机，无需关注故障
paras_EFT = {
    "RMU_ID": 0,
    "Lift_Name": 'EFT',
    "Lift_ID": 655148495849,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([-2, -1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([-2, -1]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([-2, -1]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0, 10]),
    "BrIpeak_Range": np.array([0, 10]),
    "MotIpeak_Range": np.array([0, 100]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.5,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0, 100]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 3, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }


paras_TEST = {
    "RMU_ID": 50001,
    "Lift_Name": 'TEST',
    "Lift_ID": 543940721294,
    "MissingData_Rate": MissingData_Rate,
    "Locked_SafetyI_range": np.array([0.9, 1.1]),
    "Tripped_SafetyI_range": np.array([-2, -1]),
    "Voltage_Dip_range": np.array([-2, -1]),
    "Run_SafetyI_range":np.array([-2, -1]),
    "Idle_SafetyI_range":np.array([-2, -1]),
    "Dooropen_SafetyI_range":np.array([-2, -1]),    
    "InspectionRun_range": np.array([-2, -1]),
    "InspectionStop_range": np.array([-4, -3]),
    "BrIsteady_Range": np.array([0, 10000]),
    "BrIpeak_Range": np.array([0, 10000]),
    "MotIpeak_Range": np.array([0, 12000]),
    "MotIsteady_Range": np.array([np.nan,np.nan]),
    "thres_numDoor": thres_numDoor,
    "line_Door": 0.5,
    "DoorOpen_ref": 1, # 0-用整个doorseg; 1-用safety信号再去切割出开关门信号; 2-用resv3信号再去切割出开关门信号
    "thres_DoorOpenDuration": 60,
    "DoorWaveform_type": 2,
    "DrIpeak_Range": np.array([0, 10000]),
    "RatedSpeed": 3.0,        
    "DoorError_max": 25,
    "MotorError_max": 10,
    "BrakeError_max": 15,
    # "Floor": np.array([np.nan,43.80,39.50,33.40,26.70,20.4,14.52,7.82,1.66,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    # "Floor": np.array([np.nan,43.27,39.00,32.84,26.42,20.31,14.51,7.37,1.22,np.nan,np.nan,np.nan,np.nan,np.nan]), # LG,G,1,2,3,4,5,6,7,8,9,10,11,12
    "Floor_level": np.nan, 
    "Floor_height": np.nan,
    "Position_sensor": 0, # 0-no sensor, 1-LiDAR, 2-MMU+S6000, 3-MMU+S6003 
    "thres_resv3":np.nan,
    "AI_FLAG": 0,
    "AI_Motor_Model": '', # 2023-02-22
    "AI_Brake_Model": '', # 2023-02-22
    "AI_Door_Model": '', # 2023-09-15
    "Motor_type": "AC",
    "SafetyTrip_FLAG": 1 # safety trip #1-only safety #2-safety+door #3-safety 连续trip 大于3s    
    }