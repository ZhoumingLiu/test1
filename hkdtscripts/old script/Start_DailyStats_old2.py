# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:20:02 2022

@author: chais
"""

import MySqlUtil
import MongoUtil

import datetime
import sys
import json
import pandas as pd

import os
import importlib

import numpy as np
import time

import torch
from torch import nn
import threading

from utils_lift import *
from paras_lift import *
from datetime import date, timedelta

meansObj={ 'dlcgq':['962712300013','962712300014','962712300018'],
            }
TITLES=['Brake','Safety','Resv-3']



#%%
def calModeData(dateStr):
    if dateStr=='' or len(dateStr)==0:
        # nowTime = datetime.datetime.now()
        select_day = date.today() - timedelta(1) # 计算时间的前一天
    else:
        # nowTime= datetime.datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S.%f')
        select_day = datetime.datetime.strptime(dateStr,'%Y-%m-%d').date()
    times=getTimes(select_day)

    #获取电梯设备列表
    # sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null "
    sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`,d.`c0003_field32`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null "

    #如果限制某个电梯，语句末尾增加条件 AND C0003_DEVID=10      
    # 只选择EMSD HQ L12电梯

    sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null and C0003_DEVID=655148485456"

    datas = MySqlUtil.getData(sql)
    #meansObj = json.loads(MEANS_JSON)
    #处理单个电梯设备
    MongoUtil.MongodbModule.Init()

    threads = []
    for data in datas :
        if (data[4] is  None or len(data[4]) == 0 or data[5] is  None):
            continue


        thread=threading.Thread(target=devThread,args=(data,times,select_day))
        thread.start()
        threads.append(thread)
        

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

    #%% 调用柴博士开发好的算法
    print(data[4])

    #paras = paras_EMSDHQL12
    paras= data[5]
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
def getTimes(dateStr):
    times=pd.date_range(start=str(dateStr) + ' '+'00:00:00', end=str(dateStr) + ' '+'23:59:59',
                        freq='1s').strftime('%Y-%m-%d %H:%M:%S.%f')

    return times



if __name__ =='__main__':

    # calModeData('2022-02-17 09:46:20.20') 
    t = time.perf_counter()
    calModeData('')
    print(f'coast:{time.perf_counter() - t:.8f}s')



