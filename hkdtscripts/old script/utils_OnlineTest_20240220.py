# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:04:16 2022

@author: SJ CHAI
"""

import MySqlUtil
import MongoUtil
import RedisUtil

from datetime import date,datetime,timedelta

import pandas as pd
import importlib
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy.signal import find_peaks
from math import ceil

#%% 定义。。

# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017',
#                     '962712300018','962712300022','962712300021','962712300025','962712300024','962712300028','962712300027'],
#               'jlcgq':['962712300019']}
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq',
#         'FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance']

# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017',
#                      '962712300018','962712300022','962712300021','962712300025','962712300024','962712300028','962712300027'],
#                'jlcgq':['962712300019','962715451075','962715451077']}
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq',
#         'FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance','floor','velocity']

####### update on 20240102 #########

# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017','962712300018','962712300022','962712300021','962712300025','962712300024','962712300028','962712300027'],
#              'jlcgq':['962712300019']}
             
# mmuMeans={'jlcgq':['962715451075','962715451077','962715451068','962715451069'],'dlcgq':['962715451081','962715451082']}            
             
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq','FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance','floor','velocity','height','mileage','door','level']
#%% Include MMU

# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017',
#                     '962712300018','962712300022','962712300021','962712300025','962712300024','962712300028',
#                     '962712300027'],
#               'jlcgq':['962712300019']}
             
# mmuMeans={'jlcgq':['962715451075','962715451077','962715451068','962715451069','962715451087'],
#           'dlcgq':['962715451081','962715451086']}            
             
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq','FFT-Motor Mag','FFT-Resv1 Freq',
#         'FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance','floor','velocity','height','mileage','openCloseDoorNum',
#         'door','workMode']

# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016',
#                     '962712300017','962712300018','962712300022','962712300021','962712300025',
#                     '962712300024','962712300028','962712300027'],
#               'jlcgq':['962712300019']}
             
# mmuMeans={'jlcgq':['962715451075','962715451077','962715451068','962715451069','962715451087',
#                    '962715451108','962715451091','962715451092','962715451093'],
#           'dlcgq':['962715451081','962715451086']}              
             
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq',
#         'FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag',
#         'Distance','floor','velocity','height','mileage','openCloseDoorNum','cumulativeRunNum',
#         'vibration_x','vibration_y','vibration_z','door','workMode']

# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016',
#                     '962712300017','962712300018','962712300022','962712300021','962712300025',
#                     '962712300024','962712300028','962712300027'],
#               'jlcgq':['962712300019']}
             
# mmuMeans={'jlcgq':['962715451075','962715451077','962715451076','962715451069','962715451087',
#                    '962715451079','962715451091','962715451092','962715451093'],
#               'dlcgq':['962715451081','962715451086']}              
             
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq',
#         'FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag',
#         'Distance','floor','velocity','motion','mileage','openCloseDoorNum','cumulativeRunNum',
#         'vibration_x','vibration_y','vibration_z','door','workMode']

meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017','962712300018','962712300022','962712300021','962712300025','962712300024','962712300028','962712300027'],
              'jlcgq':['962712300019']}
             
mmuMeans={'jlcgq':['962715451075','962715451077','962715451076','962715451069','962715451087','962715451079','962715451091','962715451092','962715451093'],'dlcgq':['962715451081','962715451086']}              
             
TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq','FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance','floor','velocity','motion','mileage','openCloseDoorNum','cumulativeRunNum','vibration_x','vibration_y','vibration_z','door','workMode']

#%% Include new 11 features
# meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017',
#                     '962712300018','962712300022','962712300021','962712300025','962712300024','962712300028',
#                     '962712300027'],
#              'jlcgq':['962712300019','962712315034','962712315033','962712315032','962712315031','962712315030',
#                     '962712315029','962712315028','962712315027','962712315026','962712315025','962712315024']}
             
# mmuMeans={'jlcgq':['962715451075','962715451077','962715451068','962715451069'],'dlcgq':['962715451081','962715451086']}            
             
# TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq','FFT-Motor Mag',
#         'FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance','SAD','E','P2P','Mean',
#         'Var','Skewness','Kurtosis','MAD','CF','SF','IF']


hashn=2
currthash=0
####################################
#%% 获取时间点列表(秒级)
def getTimes_secondly(dateStr):
    times=pd.date_range(start=str(dateStr) + ' '+'00:00:00', end=str(dateStr) + ' '+'23:59:59',
                        freq='1s').strftime('%Y-%m-%d %H:%M:%S.%f')

    return times

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

#获取时间点列表
def getTimes(dateStr,num):
    times=[]
    for i in range(65 * num, 5, -1):
        tempTime = dateStr + timedelta(seconds=-i)
        tempTimeStr = tempTime.strftime("%Y%m%d%H%M%S")
        for j in range(0,1000,50):
            millisecond='%03d'%j
            times.append(tempTimeStr+millisecond)
    return times


def getTimes_Hourly(dateStr):
    # end_time = datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S') + timedelta(hours=1)
    end_time = dateStr + timedelta(hours=1)
    times=pd.date_range(start=str(dateStr), end=str(end_time),
                        freq='1s').strftime('%Y-%m-%d %H:%M:%S')

    return times


def getTimes_Daily(dateStr):
    times=pd.date_range(start=str(dateStr) + ' '+'00:00:00', end=str(dateStr) + ' '+'23:59:59',
                        freq='1s').strftime('%Y-%m-%d %H:%M:%S.%f')

    return times


def Load_Build_DataFrame(dateStr, paras, Lift_ID): # 从数据库抓取和组装数据
    
    if dateStr=='' or len(dateStr)==0:
        nowTime = datetime.now()
    else:
        nowTime= datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S')
    times=getTimes(nowTime,1)
    #获取电梯设备列表
    #
    # sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null  AND clean_flag IS NOT NULL AND clean_flag!=4 "
    # sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null and C0003_DEVID=541434160882"
    # sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null and C0003_DEVID=" + str(Lift_ID)
    
    ####### update on 20240102 #########
    sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31` ,MOD(d.`C0003_DEVID`,%s"%hashn+") modv,d.api_config FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null  AND clean_flag IS NOT NULL AND clean_flag!=4 and C0003_DEVID=" + str(Lift_ID)
    ###################################
    
    datas = MySqlUtil.getData(sql)
    #meansObj = json.loads(MEANS_JSON)
    #处理单个电梯设备
    MongoUtil.MongodbModule.Init()
    RedisUtil.RedisUtil.Init()
    print(hashn)
    print(currthash)
    # threads = []
    for data in datas :
        if (data[4] is  None or len(data[4]) == 0):
            continue
        #分布式执行判断语句
        print(data)
        if(data[5]!=int(currthash)):
             continue
        print('start')    
        #print('thread')
        # thread=threading.Thread(target=devThread,args=(data,times,nowTime))
        # thread.start()
        # threads.append(thread)
        
    
        
    collName = "data_" + nowTime.strftime("%Y%m")
    
    points = []
    frameData = {}
    timeStrs = []
    for time in times:
        timeStrs.append(
            time[0:4] + "-" + time[4:6] + "-" + time[6:8] + " " + time[8:10] + ":" + time[10:12] + ":" + time[
                                                                                                         12:14] + ":" + time[
                                                                                                                        14:])
    frameData['Time'] = timeStrs
    for devtype in meansObj.keys():
        means = meansObj[devtype]
        # 获取电梯设备的电流传感器、距离传感器设备的测点
        tempPoints = getPoints(data[2], data[0], devtype, means)
        for point in tempPoints:
            points.append(point[0])


    ####### update on 20240102 #########            
    #如果是mmu电梯则追加 6个测点数据
    print('ismmu: %s' %data[6])
    if data[6]==1:     
          for devtype in mmuMeans.keys():     
            means = mmuMeans[devtype]
            # 获取电梯设备的电流传感器、距离传感器设备的测点
            tempPoints = getPoints(data[2], data[0], devtype, means)
            if tempPoints is not None:
                for point in tempPoints:
                    points.append(point[0])            
    ###################################            
            
    # 从mongodb中获取数据，拼装_id进行查询
    for index in range(len(points)):
        point = points[index]
        ids = []
        for time in times:
            ids.append(time + "_" + '%d' % point)
        mondata = MongoUtil.MongodbModule.findData("db_xgdt", collName, ids)
        framKeyData = []
        for time in times:
            if (time not in mondata):
                framKeyData.append(np.nan)
            else:
                framKeyData.append(mondata[time]['value'])
    
        frameData[TITLES[index + 1]] = framKeyData
    # 组装为pandas 数据对象
    df = pd.DataFrame(frameData)
    # print(df)    
    
    return df, data


def Load_Build_DataFrameHourly(dateStr, paras, Lift_ID): # 从数据库抓取和组装数据,一小时数据
    
    if dateStr=='' or len(dateStr)==0:
        nowTime = datetime.now()
    else:
        nowTime= datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S')
    times=getTimes_Hourly(nowTime)
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
    
        #print('thread')
        # thread=threading.Thread(target=devThread,args=(data,times,nowTime))
        # thread.start()
        # threads.append(thread)
        
    
        
    collName = "data_" + nowTime.strftime("%Y%m")
    
    points = []
    frameData = {}
    timeStrs = []
    dataTimes=[]
    for time in times:
        # timeStrs.append(time[0:4] + "-" + time[5:7] + "-" + time[8:10]+ " " + time[11:13]+ ":"+ time[14:16]+ ":" + time[17:19] + ":000")
        timeStrs.append(time)
        temp =time[0:4] + time[5:7] + time[8:10]+ time[11:13]+ time[14:16]+ time[17:19]+"000"
        dataTimes.append(temp)
    
    
    # frameData['Time'] = timeStrs
    # for devtype in meansObj.keys():
    #     means = meansObj[devtype]
    #     # 获取电梯设备的电流传感器、距离传感器设备的测点
    #     tempPoints = getPoints(data[2], data[0], devtype, means)
    #     for point in tempPoints:
    #         points.append(point[0])
    # # 从mongodb中获取数据，拼装_id进行查询
    # for index in range(len(points)):
    #     point = points[index]
    #     ids = []
    #     for time in times:
    #         ids.append(time + "_" + '%d' % point)
    #     mondata = MongoUtil.MongodbModule.findData("db_xgdt", collName, ids)
    #     framKeyData = []
    #     for time in times:
    #         if (time not in mondata):
    #             framKeyData.append(np.nan)
    #         else:
    #             framKeyData.append(mondata[time]['value'])
    
    #     frameData[TITLES[index + 1]] = framKeyData
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
    
        frameData[TITLES[index + 1]] = framKeyData
    # 组装为pandas 数据对象
    df = pd.DataFrame(frameData)
    # print(df)    
    
    return df, data


def Load_Build_DataFrameDaily(dateStr, paras, Lift_ID): # 从数据库抓取和组装数据
    
    if dateStr=='' or len(dateStr)==0:
        # nowTime = datetime.datetime.now()
        select_day = date.today() - timedelta(1) # 计算时间的前一天
    else:
        # nowTime= datetime.datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S.%f')
        select_day = datetime.strptime(dateStr,'%Y-%m-%d').date()
    times=getTimes_Daily(select_day)
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
    
        #print('thread')
        # thread=threading.Thread(target=devThread,args=(data,times,nowTime))
        # thread.start()
        # threads.append(thread)
        
    
        
    collName = "data_" + nowTime.strftime("%Y%m")
    
    points = []
    frameData = {}
    timeStrs = []
    for time in times:
        timeStrs.append(
            time[0:4] + "-" + time[4:6] + "-" + time[6:8] + " " + time[8:10] + ":" + time[10:12] + ":" + time[
                                                                                                         12:14] + ":" + time[
                                                                                                                        14:])
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
        for time in times:
            ids.append(time + "_" + '%d' % point)
        mondata = MongoUtil.MongodbModule.findData("db_xgdt", collName, ids)
        framKeyData = []
        for time in times:
            if (time not in mondata):
                framKeyData.append(np.nan)
            else:
                framKeyData.append(mondata[time]['value'])
    
        frameData[TITLES[index + 1]] = framKeyData
    # 组装为pandas 数据对象
    df = pd.DataFrame(frameData)
    # print(df)    
    
    return df, data    


    
## 序列切割函数，1s样本，无滤波，不返回car_seglist
def Data_segment_Hourly(dataset, paras, ver): 
# thres_BrakeKF为判断是否舍弃首个和最后一个segment的条件。ver是版本号，
#“1”代表用brake信号切割door cycle; “2”代表用resv3信号切割door cycle; "3"代表DC motor 并且用resv3信号切割door cycle

    dataset = dataset.reset_index()
    dataset = dataset.rename(columns={'index': 'Time'})

    

    dataset.loc[dataset['Distance'] >= 100,'Distance'] = np.nan # remove the outliers
    dataset.loc[dataset['Distance'] == 0,'Distance'] = np.nan # remove the outliers

    # dataset['Velocity'] = np.abs(dataset['Distance'].diff() / 0.05)  # calculate the abs velocity
    # dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  

    dataset['Velocity'] = np.abs(dataset['Distance'].diff())  # calculate the abs velocity
    
    # 如果brake通道的缺失率在10%以下，才做切分，否则返回空。
    if dataset['Brake'].isnull().sum()/dataset.shape[0] < 0.1:
        
        dataset = dataset.interpolate().fillna(method='bfill') # 用插值法补全缺失数据  
        
        if ver != 3:
            dataset.loc[dataset['Motor'] < 0.15,'Motor'] = 0
            dataset.loc[dataset['Resv-1'] < 0.1,'Resv-1'] = 0

        
        dataset.loc[dataset['Brake'] < 0.15,'Brake'] = 0
        dataset.loc[dataset['Door'] < 0.025,'Door'] = 0
        dataset.loc[dataset['Safety'] < 0,'Safety'] = 0
        dataset.loc[dataset['Resv-2'] < 0.1,'Resv-2'] = 0
        dataset.loc[dataset['Resv-3'] < 0,'Resv-3'] = 0
        dataset.loc[dataset['Distance'] < 0,'Distance'] = 0
        dataset.loc[dataset['Velocity'] < 0,'Velocity'] = 0
        
    
        # CarSeg_list = []
        DoorSeg_list = []
    
        dataset['carseg_flag'] = np.sign(dataset['Brake']-0.01) # -1 indicates brake close and 1 indicates brake open

        if ver==1:
            dataset['doorseg_flag'] = np.sign(0.01 - dataset['Brake']) # -1 indicates brake open and 1 indicates brake close

        else:
            dataset['doorseg_flag'] = np.sign(paras['thres_resv3']-dataset['Resv-3']) # series with -1 and 1, -1 indicates door close and 1 indicates door open


        # carseg_group = dataset[dataset['carseg_flag'] == 1].groupby((dataset['carseg_flag'] != 1).cumsum())
        doorseg_group = dataset[dataset['doorseg_flag'] == 1].groupby((dataset['doorseg_flag'] != 1).cumsum())
        
        # for k, v in carseg_group:
            
        #     CarSeg_list.append(v)  # Car motion cycle list
    
        for k, v in doorseg_group:
            
            DoorSeg_list.append(v)  # Door motion cycle list
        
            
            
       # # 删除第一个不完整的car cycle
       #  if len(CarSeg_list)>0:
       #      if CarSeg_list[0].iloc[0]['Brake'] > thres_BrakeKF:
        
       #          del (CarSeg_list[0])
                
       #  # 删除最后一个不完整的car cycle
                
       #  if len(CarSeg_list)>0:            
       #      if CarSeg_list[-1].iloc[-1]['Brake'] > thres_BrakeKF:
        
       #          del (CarSeg_list[-1])
    
                
        # 删除car segment中小于3s的cycle        
        # CarSeg_list = list(filter(lambda x: len(x)>60, CarSeg_list))
        
        # 删除door segment中为空的cycle        
        DoorSeg_list = list(filter(lambda x: len(x)>0, DoorSeg_list))
        
        # 删除door segment中小于60s的cycle        
        DoorSeg_list = list(filter(lambda x: len(x)>60, DoorSeg_list))        
        
        # 删除door segment中首尾segment     
        if len(DoorSeg_list)>0:
         
            if (DoorSeg_list[0].iloc[0]['Time'] == dataset['Time'].iloc[0]) & (DoorSeg_list[0].iloc[-1]['Time'] != dataset['Time'].iloc[-1]):
                del (DoorSeg_list[0])
                     
        if len(DoorSeg_list)>0:
      
            if (DoorSeg_list[-1].iloc[-1]['Time'] == dataset['Time'].iloc[-1]) & (len(DoorSeg_list[-1])<20): # 末尾小于20s的删除
                del (DoorSeg_list[-1])            
    
    
    else:
        # CarSeg_list = []
        DoorSeg_list = []
    
    return DoorSeg_list


def do_action_Hourly(dataset, paras, DoorSeg_list):
    event_list = []
    # CarStat_list = []
    DoorStat_list = []

    if len(DoorSeg_list) > 0:

        # door_list = []
        # door_result = []
        for i in range(len(DoorSeg_list)):
            seq = DoorSeg_list[i]        
            end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S')
             ####################### Door Rules #########################
            
            DoorStat_text,event_list = Stats_PerDoorSeg_Hourly(seq,paras,event_list)
    
            DoorStat_list.append(DoorStat_text) 


    return DoorStat_list, event_list

def Stats_PerDoorSeg_Hourly(seq, paras, event_list): # Statistic information for each long door segment/cycle, for hourly detection
    ############################################################
    ################  1. Traffic Information ###################
    ############################################################
    start_time = seq.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M:%S')
    end_time = seq.iloc[-1]['Time'].strftime('%Y-%m-%d %H:%M:%S')
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
            peaks, _ = find_peaks(seq['Door'], height = paras['line_Door'])
            num_Door = ceil(len(peaks))  # pairs of door open&closes

        elif paras['DoorWaveform_type'] == 2: # 两个波峰》一对
            peaks, _ = find_peaks(seq['Door'], height = paras['line_Door'],distance = 50)
            num_Door = ceil(len(peaks)/2)  # pairs of door open&closes

            
        elif paras['DoorWaveform_type'] == 4: # 四个波峰》一对
            peaks, _ = find_peaks(seq['Door'], height = paras['line_Door'])
            num_Door = ceil(len(peaks)/4)  # pairs of door open&closes
            
            
        elif paras['DoorWaveform_type'] == 0: # 是矩形pattern

            num_Door = len(np.argwhere(np.diff(np.sign(paras['line_Door']-seq['Door']))))/4 # number of door closing and opening pairs
            num_Door = ceil(num_Door)

    else:
        num_Door = np.nan
        
        
        
    # if num_Door > 0:
    #     DoorI_peak = seq['Door_KF'].max()
    # else:
    #     DoorI_peak = np.nan
        
    if pd.isnull(paras['thres_resv3']):
        seq_dooropen = seq.loc[(seq['Safety']>=paras['Dooropen_SafetyI_range'][0]) & (seq['Safety']<=paras['Dooropen_SafetyI_range'][1])]
        DoorOpen_Duration = len(seq_dooropen)
    else:
        DoorOpen_Duration = len(seq.loc[seq['Resv-3'] < paras['thres_resv3']]) # door open duration within each cycle    


    ############################################################
    ############  3. Event Detection based on Rules ############
    ###################################s########################
    
    if np.isnan(paras['Floor']).all() or np.isnan(seq['Distance']).all():        
        Stop_F=np.nan
    else:
        Stop_F = np.nanargmin(np.abs(seq['Distance'].iloc[0] - paras['Floor'])) - 1 # STOP floor of this cycle
    
    # # UCM     
    # if distance > 100.3 and DoorI_peak > paras['line_Door']:
    #     log_text = {
    #         "time": str(end_time),
    #         "status ID": 2.1,
    #         "event": "UCM",
    #         "description": "lift car has travelled beyond door zone (+/- 300 mm of floor level)",
    #         "floor": str(Stop_F),
    #         "delsign":0  
    #             }
    #     event_list.append(log_text)     


    # if paras['DoorOpenLong_FLAG'] == 1 and DoorOpen_Duration > paras['thres_DoorOpenDuration'] and Stop_F != 0: 
    if paras['DoorOpenLong_FLAG'] == 1 and DoorOpen_Duration > paras['thres_DoorOpenDuration']: 
        
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
        
    # if DoorI_peak < paras['DrIpeak_Range'][0] or DoorI_peak > paras['DrIpeak_Range'][1]: #2023-2-22 从非AI model里提取出来
        
    #     log_text = {
    #         "time": str(end_time),
    #         "status ID": 3.1,
    #         "event": "Door anomaly",
    #         "description": "Anomaly door motor current magnitude:" + str(round(DoorI_peak,2)) + " A",
    #         "floor": str(Stop_F),
    #         "delsign":0                  
    #             }
    #     event_list.append(log_text) 
            
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
            
    ######  Update on 2023-9-19  #####      
    DoorZero_Duration = len(seq.loc[seq['Door'] == 0]) # door current hits zero    
    if DoorZero_Duration > 10: # door 0 电流大于10s
        
        if Stop_F == -1:
            Stop_F = 'LG'
        elif Stop_F == 0:
            Stop_F = 'G'
            
        log_text = {
            "time": str(end_time),
            "status ID": 3.1,
            "event": "Door anomaly",
            "description": "Door motor current levels at " + str(round(DoorZero_Duration,2)) + " s at " + str(Stop_F) + "/F",
            "floor": str(Stop_F),
            "delsign":0  
                }
        event_list.append(log_text)  
            
            
            
    DoorStat_text = {
        'start_time':start_time,        
        'end_time':end_time,
        'duration':duration,        
        'hour':hour,
        'DoorI_peak':'NAN',
        'num_Door':num_Door,
        'DoorOpen_Duration':round(DoorOpen_Duration,2),
        'Stop_F': Stop_F
        } 
    
    DoorStat_text = {k: str(DoorStat_text[k]) if pd.isnull(DoorStat_text[k]) else DoorStat_text[k] for k in DoorStat_text }

    return DoorStat_text, event_list


# CHAI's method
def DistVel_FilterA(df,V_max,a_max):
    
    df['Dist_A']=df['Distance'].copy() # Dist_A 为距离信号滤波后的序列
    df['Vel_A']=df['Velocity'].copy() # Vel_A 为速度信号滤波后的序列
    df.loc[df['Velocity'] >= V_max,'Dist_A'] = np.nan # 找出速度序列里大于Vmax的位置，并给同样位置的距离赋值nan
    df.loc[df['Velocity'] >= V_max,'Vel_A'] = np.nan # 找出速度序列里大于Vmax的位置，并给同样位置的速度赋值nan
    df = df.interpolate().fillna(method='bfill') # 用插值法补全缺失数据nan  

    df['Acc_A'] = np.abs(df['Vel_A'].diff(periods=20))  # 计算加速度
    df.loc[df['Acc_A'] >= a_max,'Vel_A'] = np.nan # 找出速度序列里大于a_max的位置，并给同样位置的速度赋值nan
    df = df.interpolate().fillna(method='bfill') # 用插值法补全缺失数据nan  

    df['DistA_diff'] = np.abs(df['Dist_A'].diff()) # 计算相邻距离差值
    df.loc[df['DistA_diff'] >= 0.5,'Dist_A'] = np.nan # 找出距离差值序列里大于0.5的位置，并给同样位置的距离赋值nan
    df = df.interpolate().fillna(method='bfill') # 用插值法补全缺失数据nan 

    df['Dist_A'] = df['Dist_A'].resample('1s').median() # 上采样1s的中值
    df['Vel_A'] = df['Vel_A'].resample('1s').median() # 上采样1s的中值
    df = df.interpolate().fillna(method='bfill') # 用插值法补全缺失数据nan 
    return df


# Kang's method
def DistVel_FilterB(df,delta_t,a_max): # 至少取当前时刻前2s的数
    
    # delta_t = 0.5
    # a_max = 3.3
    N = len(df['Distance'])
    i = 21
    flag = 0
    k = 0
    Index_start=[]
    Index_end=[]
    df['Dist_B'] = df['Distance'].copy()
    df['Vel_raw'] = df['Distance'].copy()
    df['Vel_B'] = df['Distance'].copy()
    df['Acc_B'] = df['Distance'].copy()
    #% find the distance spike region according to the limitation of acceleration and velocity
    while i < N:
        vel = df['Dist_B'][i] - df['Dist_B'][i-20] 
        dPcal = df['Dist_B'][i] - df['Dist_B'][i-10]
        dP_max = vel*delta_t + 0.5*a_max*delta_t*delta_t 
        dP_min = vel*delta_t - 0.5*a_max*delta_t*delta_t 
        
        if not dP_min < dPcal < dP_max:
            vel = df['Dist_B'][i-1] - df['Dist_B'][i-21] 
            df['Dist_B'][i]=df['Dist_B'][i-1]+vel*delta_t*0.1
            
            if flag == 0:
                k  += 1
                
                # dataset['Index_start'][k] = i
                Index_start.append(i)
                flag = 1   
            
        else:
            if flag == 1:
            
                
                # dataset['Index_end'][k] = i
                Index_end.append(i)

                flag = 0  
                k  += 1

        
        i = i+1
                        
    #% distance interpolation        
    if len(Index_start)>0:
        for i,v in enumerate(Index_start):
            df['Dist_B'].iloc[(Index_start[i]-10):(Index_end[i]+10)]=np.nan        
            #dataset.loc['Distance',(index_start-1):(index_end+1)] = np.nan
    df['Dist_B'] = df['Dist_B'].interpolate()
    
    #% velocity calculation
    i = 20
    while i < N:
        df['Vel_raw'][i] = df['Distance'][i] - df['Distance'][i-20] 
        df['Vel_B'][i] = df['Dist_B'][i] - df['Dist_B'][i-20] 
        i = i+1
    
    i = 0    
    while i < 20:
        df['Vel_raw'][i] = df['Vel_raw'][20] 
        df['Vel_B'][i] = df['Vel_B'][20] 
        i = i+1
    
    #% acceleration calculation
    i = 20
    while i < N:
        df['Acc_B'][i] = df['Vel_B'][i] - df['Vel_B'][i-20] 
        i = i+1
        
    i = 0    
    while i < 20:
        df['Acc_B'][i] = df['Acc_B'][20]
        i = i+1
        
    # return dataset,Index_start,Index_end
    return df
 
 

def EgPlot(sequence, deviceID, fig_L, fig_W, fontsize, linewidth): # 画图
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns0 = ax1.plot(sequence['Resv-3'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='Resv-3')
    lns1 = ax1.plot(sequence['Door'], linestyle='-', 
                    linewidth=linewidth, color='#1F77B4', label='Door')
    lns2 = ax1.plot(sequence['Brake'], linestyle='-',
                    linewidth=linewidth, color='#2C9F2C', label='Brake')

    lns3 = ax1.plot(sequence['Safety'], linestyle='-', 
                    linewidth=linewidth, color='#9467BD', label='Safety')
                    
    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Door/Brake/Safety Current (A)', color=color)
#    ax1.set_ylim([0,2.0])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns4 = ax2.plot(sequence['Resv-1'], linestyle='-',
                    linewidth=linewidth, color='#bbe81a', label='Resv-1')
    
    lns5 = ax2.plot(sequence['Resv-2'], linestyle='-',
                    linewidth=linewidth, color='#FFFFCB', label='Resv-2')
    
    lns6 = ax2.plot(sequence['Motor'], linestyle='-', 
                    linewidth=linewidth, color=color, label='Motor')
    ax2.grid(False)
    ax2.set_ylabel('Motor Current (A)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
#    ax2.set_ylim([0,50])
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Set x-axis major ticks to weekly interval, on Mondays
    #ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # Format x-tick labels as %H:%m:%s
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1+lns2+lns3+lns4+lns5+lns6
    labs = [l.get_label() for l in lns]



    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, 
                framealpha=1, shadow=True, ncol = 7)
    plt.title(deviceID + '|' + str(sequence.index[0]) + '---' + str(sequence.index[-1]),pad=39)

    # duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    # ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    
    
    # # 收集所有的线条用于图例
    # lns_left = [lns0, lns1, lns2, lns3]
    # labs_left = [l.get_label() for l in lns_left]
    # lns_right = [lns4, lns5, lns6]
    # labs_right = [l.get_label() for l in lns_right]

    # # 在右侧轴上绘制左侧图例，并放置在左上角
    # ax1.legend(lns_left, labs_left, loc='upper left', bbox_to_anchor=(0, 1), fontsize=20, framealpha=1)

    # # 在右侧轴上绘制右侧图例，并放置在右上角
    # ax2.legend(lns_right, labs_right, loc='upper right', bbox_to_anchor=(1, 1), fontsize=20)

    # # 调整布局
    # plt.text(0.5, -0.08, 'Time', fontsize=20, ha='center', va='center', transform=ax1.transAxes)
    # plt.title(deviceID + '|' + str(sequence.index[0]) + '---' + str(sequence.index[-1]))

    fig.tight_layout()
    
    plt.show()
    
def EgPlot_Distance(sequence, deviceID, fig_L, fig_W, fontsize, linewidth): # 画图
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize) 

    # #############################   图1   ###################################  
    # fig, ax1 = plt.subplot(3,1,1)
    # fig, axs = plt.subplots(2)
    # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,1)
    fig, (ax1, ax3) = plt.subplots(2,1)
    
    color = 'tab:blue'
    lns0 = ax1.plot(sequence['Distance'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='Distance')

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Distance (m)', color=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns1 = ax2.plot(sequence['Velocity'], linestyle='-',
                    linewidth=linewidth, color=color, label='Velocity')
    
    ax2.grid(False)
    ax2.set_ylabel('Velocity', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1
    labs = [l.get_label() for l in lns]

    ax1.legend(lns, labs, loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 2)
    plt.title(str(sequence.index[0]) +
              '---' + str(sequence.index[-1]))
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    # plt.show()
    
    # #############################   图2   ###################################  
    # fig, ax3 = plt.subplot(3,1,2)
    color = 'tab:blue'
    lns2 = ax3.plot(sequence['Dist_A'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='Dist_A')

    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylabel('Distance (m)', color=color)
    
    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns3 = ax4.plot(sequence['Vel_A'], linestyle='-',
                    linewidth=linewidth, color=color, label='Vel_A')
    
    ax4.grid(False)
    ax4.set_ylabel('Velocity', color=color)  # we already handled the x-label with ax1
    ax4.tick_params(axis='y', labelcolor=color)
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns2+lns3
    labs = [l.get_label() for l in lns]

    ax3.legend(lns, labs, loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 2)
    plt.title('Filter A')
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax3.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    # plt.show()    
    
    plt.show()      
    
    
# 对比滤波效果
def EgPlot_Comparison(sequence, deviceID, fig_L, fig_W, fontsize, linewidth): # 画图
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize) 

    # #############################   图1   ###################################  
    # fig, ax1 = plt.subplot(3,1,1)
    # fig, axs = plt.subplots(2)
    # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,1)
    fig, (ax1, ax3, ax5) = plt.subplots(3,1)
    
    color = 'tab:blue'
    lns0 = ax1.plot(sequence['Distance'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='Distance')

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Distance (m)', color=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns1 = ax2.plot(sequence['Velocity'], linestyle='-',
                    linewidth=linewidth, color=color, label='Velocity')
    
    ax2.grid(False)
    ax2.set_ylabel('Velocity', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1
    labs = [l.get_label() for l in lns]

    ax1.legend(lns, labs, loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 2)
    plt.title(str(sequence.index[0]) +
              '---' + str(sequence.index[-1]))
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    # plt.show()
    
    # #############################   图2   ###################################  
    # fig, ax3 = plt.subplot(3,1,2)
    color = 'tab:blue'
    lns2 = ax3.plot(sequence['Dist_A'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='Dist_A')

    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylabel('Distance (m)', color=color)
    
    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns3 = ax4.plot(sequence['Vel_A'], linestyle='-',
                    linewidth=linewidth, color=color, label='Vel_A')
    
    ax4.grid(False)
    ax4.set_ylabel('Velocity', color=color)  # we already handled the x-label with ax1
    ax4.tick_params(axis='y', labelcolor=color)
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns2+lns3
    labs = [l.get_label() for l in lns]

    ax3.legend(lns, labs, loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 2)
    plt.title('Filter A')
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax3.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    # plt.show()    
    
    # #############################   图3   ###################################  
    # fig, ax5 = plt.subplot(3,1,3)
    color = 'tab:blue'
    lns4 = ax5.plot(sequence['Dist_B'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='Dist_B')

    ax5.tick_params(axis='y', labelcolor=color)
    ax5.set_ylabel('Distance (m)', color=color)
    
    ax6 = ax5.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns5 = ax6.plot(sequence['Vel_B'], linestyle='-',
                    linewidth=linewidth, color=color, label='Vel_B')
    
    ax6.grid(False)
    ax6.set_ylabel('Velocity', color=color)  # we already handled the x-label with ax1
    ax6.tick_params(axis='y', labelcolor=color)
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns4+lns5
    labs = [l.get_label() for l in lns]

    ax5.legend(lns, labs, loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 2)
    plt.title('Filter B')
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax5.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    plt.show()      
    
    
    
    
def EgPlot_Dist_Vel(sequence, deviceID, fig_L, fig_W, fontsize, linewidth): # 画图
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize) 
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    lns0 = ax1.plot(sequence['Distance'], linestyle='-', 
                    linewidth=linewidth, color=color, label='Distance (measured by LiDAR)')

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Distance (m)', color=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns1 = ax2.plot(sequence['velocity'], linestyle='-',
                    linewidth=linewidth, color=color, label='Velocity (measured by Robustel MMU)')
    
    ax2.grid(False)
    ax2.set_ylabel('velocity (m/s)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1
    labs = [l.get_label() for l in lns]

    # ax1.legend(lns, labs, loc='upper center', fancybox=True, 
    #            framealpha=1, shadow=True, ncol = 2)
    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, 
           framealpha=1, shadow=True, ncol = 2)
    # plt.title(str(sequence.index[0]) +
    #           '---' + str(sequence.index[-1]))
    plt.title(str(sequence.index[0]) +
              '---' + str(sequence.index[-1]), pad=39)    
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    plt.show()
    

    
    
def Plot_Dist_Floor(paras, sequence, deviceID, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'

    lns1 = ax1.plot(sequence['Distance'], linestyle='-', 
                    linewidth=linewidth, color='#1F77B4', label='Distance (measured by LiDAR)')

                    

    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    # new_ticks = np.array([44.77,39.03,35.71,31.93,26.19,19.26,13.16,7.49,1.25]) #L8
    # new_ticks = np.array([63.17,57.37,50.21,44.4,37.51,31.44,25.69,19.84]) #L11
    height_ticks = np.array(paras['Floor_height'])
    level_ticks = paras['Floor_level']
    ax1.set_yticks(height_ticks)
    
    heightticks_string = [str(item) for item in height_ticks]
    levelticks_string = ['('+ item + ')' for item in level_ticks]
    combine_ticks = [i + j for i, j in zip(heightticks_string, levelticks_string)]
    ax1.set_yticklabels(combine_ticks) 
    
    ax1.invert_yaxis()

    ax1.set_ylabel('Distance from the shaft roof (m)', color=color)
#    ax1.set_ylim([0,2.0])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns2 = ax2.plot(sequence['floor'], linestyle='-',
                    linewidth=linewidth, color=color, label='Floor (measured by Robustel MMU)')
    

    ax2.grid(False)
    ax2.set_ylabel('Floor', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    index_list = [i for i in range(len(level_ticks))]
    new_ticks = np.array(index_list)
    ax2.set_yticks(new_ticks)
    ax2.set_yticklabels(level_ticks)
#    ax2.set_ylim([0,50])
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Set x-axis major ticks to weekly interval, on Mondays
    #ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # Format x-tick labels as %H:%m:%s
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, bbox_to_anchor=(-0.05, 0.05), loc='upper left',
    #           fancybox=True, framealpha=1,shadow=True, borderpad=1)
    # ax1.legend(lns, labs, ncol=6, loc='upper center')
    # ax1.legend(lns, labs, ncol=4, loc='upper center')

    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 2)
    plt.title(str(sequence.index[0]) +
              '---' + str(sequence.index[-1]), pad=39)
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    plt.show()    
    
    
    
    
def Plot_Brake_Floor(paras, sequence, deviceID, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'

    lns1 = ax1.plot(sequence['Brake'], linestyle='-', 
                    linewidth=linewidth, color='#1F77B4', label='Brake Current(measured by RMU)')

                    

    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Brake Current (A)', color=color)
#    ax1.set_ylim([0,2.0])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns2 = ax2.plot(sequence['floor'], linestyle='-',
                    linewidth=linewidth, color=color, label='Floor (measured by Robustel MMU)')
    

    ax2.grid(False)
    ax2.set_ylabel('Floor', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    level_ticks = paras['Floor_level']
    index_list = [i for i in range(len(level_ticks))]
    new_ticks = np.array(index_list)
    ax2.set_yticks(new_ticks)
    ax2.set_yticklabels(level_ticks)
#    ax2.set_ylim([0,50])
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Set x-axis major ticks to weekly interval, on Mondays
    #ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # Format x-tick labels as %H:%m:%s
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, bbox_to_anchor=(-0.05, 0.05), loc='upper left',
    #           fancybox=True, framealpha=1,shadow=True, borderpad=1)
    # ax1.legend(lns, labs, ncol=6, loc='upper center')
    # ax1.legend(lns, labs, ncol=4, loc='upper center')

    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 2)
    plt.title(str(sequence.index[0]) +
              '---' + str(sequence.index[-1]), pad=39)
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    plt.show()    
        
    
def EgPlot_RawVSKF(seq1,seq2, deviceID, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns0 = ax1.plot(seq1, linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label=seq1.name)
    lns1 = ax1.plot(seq2, linestyle='-', 
                    linewidth=linewidth, color='#1F77B4', label=seq2.name)

                    

    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Current (A)', color=color)
#    ax1.set_ylim([0,2.0])

#    ax2.set_ylim([0,50])
    
    
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Set x-axis major ticks to weekly interval, on Mondays
    #ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # Format x-tick labels as %H:%m:%s
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1
    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, bbox_to_anchor=(-0.05, 0.05), loc='upper left',
    #           fancybox=True, framealpha=1,shadow=True, borderpad=1)
    # ax1.legend(lns, labs, ncol=6, loc='upper center')
    # ax1.legend(lns, labs, ncol=4, loc='upper center')

    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 7)
    plt.title(str(seq1.index[0]) +
              '---' + str(seq1.index[-1]), pad=39)
    duration = (seq1.index[-1] - seq1.index[0]).total_seconds()
    ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    plt.show()
    
    
def ArupPlot(sequence, deviceID, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns0 = ax1.plot(sequence['Resv-3'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='Resv-3')
    lns1 = ax1.plot(sequence['Door'], linestyle='-', 
                    linewidth=linewidth, color='#1F77B4', label='Door')
    lns2 = ax1.plot(sequence['Brake'], linestyle='-',
                    linewidth=linewidth, color='#2C9F2C', label='Brake')

    lns3 = ax1.plot(sequence['Safety'], linestyle='-', 
                    linewidth=linewidth, color='#9467BD', label='Safety')
                    

    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Door/Brake/Safety Current (A)\n Velocity (m/s)', color=color)
#    ax1.set_ylim([0,2.0])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'

    
    lns4 = ax2.plot(sequence['Motor'], linestyle='-', 
                    linewidth=linewidth, color=color, label='Motor')
    ax2.grid(False)
    ax2.set_ylabel('Motor Current (A)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
#    ax2.set_ylim([0,50])
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Set x-axis major ticks to weekly interval, on Mondays
    #ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # Format x-tick labels as %H:%m:%s
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, bbox_to_anchor=(-0.05, 0.05), loc='upper left',
    #           fancybox=True, framealpha=1,shadow=True, borderpad=1)
    # ax1.legend(lns, labs, ncol=6, loc='upper center')
    # ax1.legend(lns, labs, ncol=4, loc='upper center')

    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, 
               framealpha=1, shadow=True, ncol = 5)
    plt.title('Current Signals:' + str(sequence.index[0]) +
              '---' + str(sequence.index[-1]), pad=39)
    duration = (sequence.index[-1] - sequence.index[0]).total_seconds()
    ax1.set_xlabel( deviceID + '-' + '    [ Duration: ' + str(duration) + ' s ]')
    plt.show()
    
def FFTMag_plot(sequence, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns0 = ax1.plot(sequence['FFT-Motor Mag'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='FFT-Motor_U Mag')
    lns1 = ax1.plot(sequence['FFT-Resv1 Mag'], linestyle='-', 
                    linewidth=linewidth, color='#1F77B4', label='FFT-Motor_V Mag')
    lns2 = ax1.plot(sequence['FFT-Resv2 Mag'], linestyle='-',
                    linewidth=linewidth, color='#2C9F2C', label='FFT-Motor_W Mag')

    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('FFT Magnitude (A)', color=color)
#    ax1.set_ylim([0,2.0])

    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1+lns2
    labs = [l.get_label() for l in lns]
#    ax1.legend(lns, labs, bbox_to_anchor=(1.05, 1.0), loc='upper left',
#               fancybox=True, framealpha=1,shadow=True, borderpad=1)
    # ax1.legend(lns, labs, ncol=6, loc='upper center')
    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, framealpha=1, shadow=True, ncol=3)
    plt.title('FFT:' + str(sequence.index[0]) +
              '---' + str(sequence.index[-1]),pad=39)
    ax1.set_xlabel('Time')
    plt.show()


def FFTFreq_plot(sequence, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns0 = ax1.plot(sequence['FFT-Motor Freq'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='FFT-Motor_U Freq')
    lns1 = ax1.plot(sequence['FFT-Resv1 Freq'], linestyle='-', 
                    linewidth=linewidth, color='#1F77B4', label='FFT-Motor_V Freq')
    lns2 = ax1.plot(sequence['FFT-Resv2 Freq'], linestyle='-',
                    linewidth=linewidth, color='#2C9F2C', label='FFT-Motor_W Freq')

    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('FFT Frequency (Hz)', color=color)
#    ax1.set_ylim([0,2.0])

    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1+lns2
    labs = [l.get_label() for l in lns]
#    ax1.legend(lns, labs, bbox_to_anchor=(1.05, 1.0), loc='upper left',
#               fancybox=True, framealpha=1,shadow=True, borderpad=1)
    # ax1.legend(lns, labs, ncol=6, loc='upper center')
    ax1.legend(lns, labs, bbox_to_anchor=(0.5, 1.14), loc='upper center', fancybox=True, framealpha=1, shadow=True, ncol=3)
    plt.title('FFT:' + str(sequence.index[0]) +
              '---' + str(sequence.index[-1]),pad=39)
    ax1.set_xlabel('Time')
    plt.show()

def FFTFreqMag_plot(sequence, fig_L, fig_W, fontsize, linewidth):
   
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns0 = ax1.plot(sequence['FFT-Motor Freq'], linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label='FFT-Motor_U Freq')

    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Frequency (Hz)', color=color)
#    ax1.set_ylim([0,2.0])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    lns1 = ax2.plot(sequence['FFT-Motor Mag'], linestyle='-', 
                    linewidth=linewidth, color=color, label='FFT-Motor_U Mag')
    ax2.grid(False)
    ax2.set_ylabel('FFT Magnitude (A)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
#    ax2.set_ylim([0,50])
    
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Set x-axis major ticks to weekly interval, on Mondays
    #ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    # Format x-tick labels as %H:%m:%s
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # set all labels in one legend
    lns = lns0+lns1
    labs = [l.get_label() for l in lns]
#    ax1.legend(lns, labs, bbox_to_anchor=(1.05, 1.0), loc='upper left',
#               fancybox=True, framealpha=1,shadow=True, borderpad=1)
    ax1.legend(lns, labs, ncol=2, loc='upper center')

    plt.title('FFT:' + str(sequence.index[0]) +
              '---' + str(sequence.index[-1]),pad=39)
    ax1.set_xlabel('Time')
    plt.show()
    
def EgPlot_SingleFeature(df,feature,deviceID, fig_L, fig_W, fontsize, linewidth):
    sns.set(rc={'figure.figsize':(fig_L, fig_W)})
    sns.set(font_scale=fontsize)  
    fig, ax1 = plt.subplots()
    color = '#0c0c0d'
    seq = df[feature]
    lns0 = ax1.plot(seq, linestyle='-', 
                    linewidth=linewidth, color='#0c0c0d', label=feature)


                    
    #ax1.grid(False)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel(feature, color=color)
#    ax1.set_ylim([0,2.0])

#    ax2.set_ylim([0,50])
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));

    fig.tight_layout()  # otherwise the right y-label is slightly clipped


    ax1.legend()
    plt.title(deviceID + '|' + 'Door Motor Current ' + feature + ' Sequence')
    plt.show()