# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 18:29:33 2022

@author: SJ CHAI
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:20:02 2022

@author: chais
"""
import os

os.chdir("C:/Users/SJ CHAI/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts/")
# os.chdir("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/runErgatianScript/hkdtscripts/")

import MySqlUtil
import MongoUtil

import datetime
import sys
import json
import pandas as pd

import importlib

import numpy as np
import time

# import torch
# from torch import nn
import threading

# from utils_lift import *
# from paras_lift import *
from datetime import date, timedelta
from utils_OnlineTest import *



# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017',
#                     '962712300018','962712300022','962712300021','962712300025','962712300024','962712300028','962712300027'],
#               'jlcgq':['962712300019','962715451075','962715451077']}
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq',
#         'FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance','floor','velocity']

# meansObj={ 'jlcgq':['962712300019','962715451075','962715451077']}
## TITLES=['floor','velocity','Distance']
## TITLES=['Distance','floor','velocity']
# TITLES=['Time','velocity','floor','Distance']



meansObj={ 'dlcgq':['962712300013'],'jlcgq':['962712300019','962715451075','962715451077']}
TITLES=['Time','Brake','velocity','floor','Distance']
#%% 获取电梯相应设备数据测点点号
def getPoints(stationId,devId,devType,means):
    meanSql=""
    for index in range(len(means)):
        if index==0:
            meanSql+=means[index]
        else :
            meanSql+=","+means[index]
    sql="SELECT p.C0110_POINT_ID,p.C0110_FARM_TERM_NO FROM sys_m_point p,sys_m_device d WHERE p.C0003_DEVID=d.C0003_DEVID and p.C0002_STATION_NO="+'%d'%stationId+" and d.c0003_parent_id="+devId+" and d.C0003_DEVTYPE='"+devType+"' and p.C0029_MEASTYPEID in ("+meanSql+") order by p.sort"
    #print(sql)
    return  MySqlUtil.getData(sql)

#%% 获取时间点列表
def getTimes_secondly(dateStr):
    times=pd.date_range(start=str(dateStr) + ' '+'00:00:00', end=str(dateStr) + ' '+'23:59:59',
                        freq='1s').strftime('%Y-%m-%d %H:%M:%S.%f')

    return times


#%% 获取该日的数据 选择电梯并计算

dateStr = '2023-12-06 14:20:44'

# dateStr = ''

from paras_lift import paras_EMSDHQL11

paras = paras_EMSDHQL11

Lift_ID = paras['Lift_ID']

# select_day = datetime.strptime(dateStr,'%Y-%m-%d').date()
select_day = pd.to_datetime(dateStr).date()
times=getTimes_secondly(select_day.strftime('%Y-%m-%d')) #获取1s间隔的时间index
#获取电梯设备列表
#
# sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null  AND clean_flag IS NOT NULL AND clean_flag!=4 "
# sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null and C0003_DEVID=541434160882"
sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null and C0003_DEVID=" + str(Lift_ID)

datas = MySqlUtil.getData(sql)
#meansObj = json.loads(MEANS_JSON)
#处理单个电梯设备
MongoUtil.MongodbModule.Init()
RedisUtil.RedisUtil.Init()

threads = []
for data in datas :
    if (data[4] is  None or len(data[4]) == 0):
        continue
    
collName = "data_" + select_day.strftime("%Y%m")

points = []
frameData = {}
timeStrs = []
dataTimes=[]

for time in times:
    timeStrs.append(time)
    temp =time[0:4] + time[5:7] + time[8:10]+ time[11:13]+ time[14:16]+ time[17:19]+"000"
    dataTimes.append(temp)
    #timeStrs.append( time[0:4] + "-" + time[4:6] + "-" + time[6:8] + " " + time[8:10] + ":" + time[10:12] + ":" + time[12:14] + ":" + time[ 14:])

frameData['Time'] = timeStrs
for devtype in meansObj.keys():
    means = meansObj[devtype]
    # 获取电梯设备的电流传感器、距离传感器设备的测点
    tempPoints = getPoints(data[2], data[0], devtype, means)
    for point in tempPoints:
        points.append(point[0])
# 从mongodb中获取数据，拼装_id进行查询
for index in range(len(points)):
    point = points[index]
    ids = []
    for time in dataTimes:
        ids.append(time + "_" + '%d' % point)

    mondata = MongoUtil.MongodbModule.findData("db_xgdt", collName, ids)
    framKeyData = []
    for time in dataTimes:
        if (time not in mondata):
            framKeyData.append(np.nan)
        else:
            framKeyData.append(mondata[time]['value'])

    frameData[TITLES[index+1]] = framKeyData
# 组装为pandas 数据对象
df = pd.DataFrame(frameData)


#%% 分析！！！
dataset = df.copy()
dataset['Time'] = pd.to_datetime(dataset['Time'],format='%Y-%m-%d %H:%M:%S')
dataset = dataset.set_index('Time') # set datetime-based index

dataset = dataset.rename(columns={"floor": "velocity", "velocity": "floor"})
dataset = dataset.astype(float)

##1.计算floor=NAN的长度和丢包率
# MissingFloor = len(dataset.loc[dataset['floor']==float('nan')])/len(dataset)         

MissingFloor = np.isnan(dataset['velocity']).values.sum()/len(dataset) 
print(MissingFloor)

##2.查看连续NAN的数量 

##3.画velocity和distance的图

seq = dataset.loc['2023-12-06 13:02:00':'2023-12-06 13:07:00']
seq['velocity'] = seq['velocity'].interpolate() #补齐velocity，用均值代替

EgPlot_Dist_Vel(seq, 'EMSDHQ_L11', 12, 5, 1.5, 2)


#4.画Floor和distance的图
seq['floor'] = seq['floor'].ffill() #补齐楼层，用最近值代替空值

seq.loc[seq['floor'] == -1,'floor'] = 0
seq.loc[seq['floor'] == -2,'floor'] = -1
# seq.loc[seq['floor'] == 'NAN','floor'] = np.nan
seq['floor'] = seq['floor'].astype(float) 

Plot_Dist_Floor(paras, seq, 'EMSDHQ_L11', 12, 5, 1.5, 2)
                                                                   

#5.画brake和floor的图
Plot_Brake_Floor(paras, seq, 'EMSDHQ_L11', 12, 5, 1.5, 2)


#%%
df, data = Load_Build_DataFrameDaily(dateStr, paras, Lift_ID)
# module = importlib.import_module('devpy' + data[4])
# operation_class = getattr(module, "runMethod")
# result,CarSeg_Stats,DoorSeg_Stats = operation_class(df)
# print(result)    

df['Time'] = pd.to_datetime(df['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
df = df.set_index('Time') 

# df['Velocity'] = np.abs(df['Distance'].diff(periods=20))  # calculate the abs velocity
# df = df.interpolate().fillna(method='bfill') # 用插值法补全缺失数据 

#%%
# def calModeData(dateStr):
#     if dateStr=='' or len(dateStr)==0:
#         # nowTime = datetime.datetime.now()
#         select_day = date.today() - timedelta(1) # 计算时间的前一天
#     else:
#         # nowTime= datetime.datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S.%f')
#         select_day = datetime.datetime.strptime(dateStr,'%Y-%m-%d').date()
#     times=getTimes(select_day)

#     #获取电梯设备列表
#     #如果限制某个电梯，语句末尾增加条件 AND C0003_DEVID=10  
#     # sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null "
#     # 只选择EMSD HQ L12电梯
#     # sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`,d.`c0003_field32`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null "
    
#     sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`,d.`c0003_field32`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null and C0003_DEVID=655148495847"

#     datas = MySqlUtil.getData(sql)
#     #meansObj = json.loads(MEANS_JSON)
#     #处理单个电梯设备
#     MongoUtil.MongodbModule.Init()

#     threads = []
#     for data in datas :
#         if (data[4] is  None or len(data[4]) == 0 or data[5] is  None):
#             continue


#         thread=threading.Thread(target=devThread,args=(data,times,select_day))
#         thread.start()
#         threads.append(thread)
        

#%%
def  devThread(data,times,select_day):
    collName = "data_" + select_day.strftime("%Y%m")

    points = []
    frameData = {}
    timeStrs = []
    dataTimes=[]
    for time in times:
        timeStrs.append(time)
        temp =time[0:4] + time[5:7] + time[8:10]+ time[11:13]+ time[14:16]+ time[17:19]+"000"
        dataTimes.append(temp)
        #timeStrs.append( time[0:4] + "-" + time[4:6] + "-" + time[6:8] + " " + time[8:10] + ":" + time[10:12] + ":" + time[12:14] + ":" + time[ 14:])

    frameData['Time'] = timeStrs
    for devtype in meansObj.keys():
        means = meansObj[devtype]
        # 获取电梯设备的电流传感器、距离传感器设备的测点
        tempPoints = getPoints(data[2], data[0], devtype, means)
        for point in tempPoints:
            points.append(point[0])
    # 从mongodb中获取数据，拼装_id进行查询
    for index in range(len(points)):
        point = points[index]
        ids = []
        for time in dataTimes:
            ids.append(time + "_" + '%d' % point)

        mondata = MongoUtil.MongodbModule.findData("db_xgdt", collName, ids)
        framKeyData = []
        for time in dataTimes:
            if (time not in mondata):
                framKeyData.append(np.nan)
            else:
                framKeyData.append(mondata[time]['value'])

        frameData[TITLES[index]] = framKeyData
    # 组装为pandas 数据对象
    df = pd.DataFrame(frameData)
    print(df)
    print(df.dtypes)

    #%% 调用柴博士开发好的算法
    print(data[4])
    # df.to_csv('out.csv')

    # paras = paras_EMSDHQL11
    # paras= data[5]
    paras= eval(data[5])
    print(paras)
    Daily_paras = DailyParas_Calculate(df,paras)
    print(Daily_paras)

    
    #%% 对结果进行解析存储
    # insertSql = "INSERT INTO bs_daily_paras(floor_id,lift_id,data_date,duration_offline,duration_lock,duration_trip,duration_dip,duration_run,duration_idle,duration_dooropen,duration_inspection,avg_safetyI_run,avg_safetyI_idle," \
    #             "avg_safetyI_dooropen ) VALUES(" + '%s' % \
    #             data[2] + "," + '%s' % data[0] + ",'" + '%s' % Daily_paras['Data'].strftime("%Y-%m-%d") + "','" + '%s' % Daily_paras['duration_offline']+ "','" + '%s' % Daily_paras['duration_lock']+ "','" + '%s' % Daily_paras['duration_trip']+ "','" \
    #             + '%s' % Daily_paras['duration_dip']+ "','" + '%s' % Daily_paras['duration_run']+ "','" + '%s' % Daily_paras['duration_idle']+ "','"+  '%s' % Daily_paras['duration_dooropen']+ "','" + '%s' % Daily_paras['duration_inspection']+ "','" + '%s' % Daily_paras['avg_safetyI_run']+ \
    #            "','" + '%s' % Daily_paras['avg_safetyI_idle']+ "','" + '%s' % Daily_paras['avg_safetyI_dooropen']+"')"
    # print(insertSql)
    # deleteSql="delete from bs_daily_paras where floor_id=%s" %data[2] + " and lift_id='"+"%s"%data[0]+"' and data_date='"+"%s"%Daily_paras['Data'].strftime("%Y-%m-%d")+"'"
    # print(deleteSql)
    # MySqlUtil.executSql(deleteSql)
    # saveRe = MySqlUtil.executSql(insertSql)


    


if __name__ =='__main__':

    # calModeData('2022-02-17 09:46:20.20') 
    t = time.perf_counter()
    calModeData('')
    print(f'coast:{time.perf_counter() - t:.8f}s')



