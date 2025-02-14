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


meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017','962712300018','962712300022','962712300021','962712300025','962712300024','962712300028','962712300027'],
             'jlcgq':['962712300019']}
TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq','FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance']


def calModeData(dateStr):
    if dateStr=='' or len(dateStr)==0:
        nowTime = datetime.datetime.now()
    else:
        nowTime= datetime.datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S.%f')
    times=getTimes(nowTime,1)
    #获取电梯设备列表
    #如果限制某个电梯，语句末尾增加条件 AND C0003_DEVID=10  
    # sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null "
    # 只选择EMSD HQ L12电梯
    sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null and C0003_DEVID=10"
   
    datas = MySqlUtil.getData(sql)
    #meansObj = json.loads(MEANS_JSON)
    #处理单个电梯设备
    MongoUtil.MongodbModule.Init()

    threads = []
    for data in datas :
        if (data[4] is  None or len(data[4]) == 0):
            continue

        print('thread')
        thread=threading.Thread(target=devThread,args=(data,times,nowTime))
        thread.start()
        threads.append(thread)
        


def  devThread(data,times,nowTime):
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
    # 数据清洗，清洗掉nan数据
    # newdf =df.dropna(subset=['Brake', 'Door'], inplace=True)
    # if newdf is None :
    #    continue
    # 调用柴博士开发好的算法
    print(data[4])
    module = importlib.import_module('devpy' + data[4])
    operation_class = getattr(module, "runMethod")
    result = operation_class(df)
    # result = run_4008636142EMSDHQL6.runMethod(newdf)
    print(result)
    # 对结果进行解析存储
    insertSql = "INSERT INTO bs_lift_status(floor_id,lift_id,last_status,period_start,period_end,post_time) VALUES(" + '%s' % \
                data[2] + "," + '%s' % data[0] + ",'" + '%s' % result['last_status'] + "','" + '%s' % result['period'][
                    'start'] + "','" + '%s' % result['period']['end'] + "','" + '%s' % result['post_time'] + "')"
    # saveRe = MySqlUtil.saveData(insertSql) # 入库程序
    eventVal = []
    # for event in result['event_list']:
    #     tup = (saveRe['lastrowid'], event['description'], event['status ID'], event['time'])
    #     eventVal.append(tup)
    # insertEvent = "INSERT INTO bs_lift_events(result_id,event_desc,status_id,event_time) VALUES(%s,%s,%s,%s)"
    # if (len(eventVal) > 0):
    #     MySqlUtil.saveBatchData(insertEvent, eventVal)  # 入库程序
    

#获取电梯相应设备数据测点点号
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
        tempTime = dateStr + datetime.timedelta(seconds=-i)
        tempTimeStr = tempTime.strftime("%Y%m%d%H%M%S")
        for j in range(0,1000,50):
            millisecond='%03d'%j
            times.append(tempTimeStr+millisecond)
    return times



if __name__ =='__main__':

    # calModeData('2021-12-11 13:25:00.20') 
    t = time.perf_counter()
    calModeData('')
    print(f'coast:{time.perf_counter() - t:.8f}s')



