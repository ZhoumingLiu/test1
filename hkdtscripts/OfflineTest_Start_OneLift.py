# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:53:14 2021

@author: chais
"""


#%% Offline test main code

import os


import pandas as pd

# os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts")
# from utils_lift import *

# os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/Utils/")
os.chdir("C:/Users/SJ CHAI/Dropbox/Spyder_Dropbox/Elevator EMSD code/Utils/")

from utils_ErgatianLifts import *

# os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts/")
os.chdir("C:/Users/SJ CHAI/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts/")

from paras_lift import paras_QEBL8
# from utils_lift_QEBL8 import *
from utils_lift import *
from devpy import run_15_QEBL8

#%% Load the dataset for single file (单个csv文件读取)



# dataset = pd.read_csv('C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/DataFiles/L1_2024-05-07_new features.csv',index_col = False)
# dataset = pd.read_csv('C:/Users/SJ CHAI/Dropbox/Spyder_Dropbox/Elevator EMSD code/DataFiles/L1_2024-05-07_new features.csv',index_col = False)
dataset = pd.read_csv('D:/Dropbox/Spyder_Dropbox/Elevator EMSD code/DataFiles/L1_2024-05-07_new features.csv',index_col = False)

dataset['Time'] = pd.to_datetime(dataset['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')


dataset = dataset.astype({'Motor':'float','Brake':'float','Safety':'float','Door':'float','Resv-1':'float','Resv-2':'float','Resv-3':'float','Distance':'float'})

#%%
############################## %% Select a period and run (for single file) ##############################
seq = dataset.loc[(dataset['Time']>='2024-05-07 13:54:25') & (dataset['Time']<='2024-05-07 13:54:27')]

# Run the main code
Result,CarSeg_Stats,DoorSeg_Stats = run_15_QEBL8.runMethod(seq)

print(Result)
print(CarSeg_Stats)
print(DoorSeg_Stats)


#%% Load the dataset for single file (pkl文件)

carseg_raw = pd.read_pickle('C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/DataFiles/KF_QMH_CarSegList_202210') # car motion

######### Plot motor current cluster for different floors traveled ###########

# idx = df_featureV2[(df_featureV2['F_Travelled'] == 6) & (df_featureV2['Dir_Travelled']=='upward')].index 
# carseg_slice = [carseg_raw[i] for i in idx]
import matplotlib.pyplot as plt

for i in range(len(carseg_raw)):
    
    if 300<len(carseg_raw[i])<350:
        
        plt.plot(carseg_raw[i]['Motor_KF'].values)
plt.ylabel('Current (A)')
plt.xlabel('Readings')
plt.show()

#%% Read multiple CSV files

# import glob
# import sys
# from tqdm import tqdm
# # 加载该电梯的关键参数
# # paras = paras_EMSDHQL12


# directoryPath = "G:/BaiduNetdiskDownload/EMSD/EMSDHQ_L12/202212/"

# progress=0 # progress flag
# nfiles = len(glob.glob(directoryPath+'*.csv'))
# # Trip_list = []
# # SpeedSpike_list = []
# # SpikeTime_list = []

# for file_name in tqdm(glob.glob(directoryPath+'*.csv')):
    
#     dataset_raw = pd.read_csv(file_name,index_col = False )
#     dataset = dataset_raw.copy()
    
#     dataset['Time'] = pd.to_datetime(dataset['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
#     # dataset['Time'] = pd.to_datetime(dataset['Time'],format='%Y-%m-%d %H:%M:%S:%f')
#     # dataset['Time'] = dataset['Time'].dt.strftime("%Y-%m-%d %H:%M:%S")
#     dataset = dataset.astype({'Motor':'float','Brake':'float','Safety':'float','Door':'float','Resv-1':'float','Resv-2':'float','Resv-3':'float','Distance':'float'})
    
#     i = 0
#     while i<len(dataset):
#         # print(i)

#         seq = dataset.iloc[i : (i+72000)]
        
        
#         # Run the main code
#         Result,_,_ = run_21_EMSDHQL12.runMethod(seq) 
        
        
#         txt_file = open("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/DataFiles/EMSD HQ L12_2022-12.txt", "a", encoding="utf-8")  # 以写的格式打开先打开文件
#         txt_file.write(str(Result))
#         txt_file.write("\n")
#         txt_file.close()
        
        
#         i = i+72000


    # dataset = dataset.set_index('Time') # set datetime-based index
    # dataset = dataset.astype(float)
    
#%% plot

# Plot
deviceID = 'EMSDHQL12'
# segment = DoorSeg_list[0].set_index('Time')
seq = seq.set_index('Time')
EgPlot(seq,deviceID, 12, 5, 1.5, 2)
















