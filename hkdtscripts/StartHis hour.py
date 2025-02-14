import MySqlUtil
import MongoUtil
import RedisUtil

from datetime import datetime,timedelta
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
from concurrent.futures import ThreadPoolExecutor

from utils_lift import *


meansObj={ 'dlcgq':['962712300012','962712300013','962712300014','962712300015','962712300016','962712300017','962712300018','962712300022','962712300021','962712300025','962712300024','962712300028','962712300027'],
             'jlcgq':['962712300019']}
TITLES=['Time','Motor','Brake','Safety','Door','Resv-1','Resv-2','Resv-3','FFT-Motor Freq','FFT-Motor Mag','FFT-Resv1 Freq','FFT-Resv1 Mag','FFT-Resv2 Freq','FFT-Resv2 Mag','Distance']

hashn=2
currthash=0

def calModeData(dateStr):
    if dateStr=='' or len(dateStr)==0:
        nowTime = datetime.now()
    else:
        nowTime= datetime.strptime(dateStr,'%Y-%m-%d %H:%M:%S')
    times=getTimes(nowTime,1)
    #获取电梯设备列表
    #
    sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null  AND clean_flag IS NOT NULL AND clean_flag!=4 "
    datas = MySqlUtil.getData(sql)
    #meansObj = json.loads(MEANS_JSON)
    #处理单个电梯设备


    threads = []
    #pool = ThreadPoolExecutor(max_workers=10)
    for data in datas :
        if (data[4] is  None or len(data[4]) == 0):
            continue
        #分布式执行判断语句
        #print(data)


        #print('thread')
        #thread=threading.Thread(target=devThread,args=(data,times,nowTime))
        #thread.start()
        #threads.append(thread)

        #future1 = pool.submit(devThread, data,times,nowTime)
        devThread(data,times,nowTime)
    #pool.shutdown()
        


def  devThread(data,times,nowTime):
    collName = "data_" + nowTime.strftime("%Y%m")
    MongoUtil.MongodbModule.Init()
    RedisUtil.RedisUtil.Init()
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
    #print(data[4])
    module = importlib.import_module('devpy' + data[4])
    operation_class = getattr(module, "runMethod")
    result,CarSeg_Stats,DoorSeg_Stats = operation_class(df)
    # result = run_4008636142EMSDHQL6.runMethod(newdf)
    print(result)
    # print(CarSeg_Stats)
    # 对结果进行解析存储
    
    
    # saveResult(nowTime,data[2],data[0],result,CarSeg_Stats,DoorSeg_Stats)
    
    
    #print(DoorSeg_Stats)

    
    # return df

def saveResult(nowTime,floorId,liftId,result,CarSeg_Stats,DoorSeg_Stats):
    #保存result
    isReset=0
    if result['last_status'] == 0:
        isReset=1
    key='his_status:%s'%liftId
    oriStatus=RedisUtil.RedisUtil.getStatus(key)


    if not oriStatus:


        sql = "INSERT INTO bs_lift_status(floor_id,lift_id,last_status,period_start,period_end,post_time,is_reset) VALUES(" + '%s' % \
                    floorId + "," + '%s' % liftId + ",'" + '%s' % result['last_status'] + "','" + '%s' % \
                    result['period']['start'] + "','" + '%s' % result['period']['end'] + "','" + '%s' % result['post_time'] +"',%s"%isReset+")"
        saveRe = MySqlUtil.executSql(sql)

        oriStatus={}
        oriStatus['floor_id']=floorId
        oriStatus['lift_id'] = liftId
        oriStatus['last_status'] = result['last_status']
        oriStatus['period_start'] = result['period']['start'].strftime('%Y-%m-%d %H:%M:%S:%f')
        oriStatus['period_end'] = result['period']['end'].strftime('%Y-%m-%d %H:%M:%S:%f')
        oriStatus['post_time'] = result['post_time'].strftime('%Y-%m-%d %H:%M:%S:%f')
        oriStatus['result_id'] = saveRe['lastrowid']

        oriStatus['is_reset'] = isReset
        RedisUtil.RedisUtil.setStatus(key,oriStatus)

    else:

        if oriStatus['last_status'] == "%s"%result['last_status'] and result['last_status'] != 0:
            sql="update bs_lift_status set is_reset=0,period_end='"+'%s' % result['period']['end'] +"',post_time='"+'%s' % result['post_time']+"' where result_id="+oriStatus['result_id']
        elif oriStatus['last_status'] !='0' and result['last_status'] == 0:
            sql = "update bs_lift_status set period_end='" + '%s' % result['period']['end'] + "',post_time='" + '%s' % \
                  result['post_time'] + "',is_reset=%s"%isReset+' where result_id=' + oriStatus['result_id']
        elif oriStatus['last_status'] == '0' and result['last_status'] == 0:
            sql = "update bs_lift_status set post_time='" + '%s' % \
                  result['post_time'] + "' where result_id=" + oriStatus['result_id']

        else:
            sql = "INSERT INTO bs_lift_status(floor_id,lift_id,last_status,period_start,period_end,post_time,is_reset) VALUES(" + '%s' % \
                  floorId + "," + '%s' % liftId + ",'" + '%s' % result['last_status'] + "','" + '%s' % \
                  result['period']['start'] + "','" + '%s' % result['period']['end'] + "','" + '%s' % result['post_time'] + "',%s" %isReset + ")"
            oriStatus['period_start'] = result['period']['start'].strftime('%Y-%m-%d %H:%M:%S:%f')
        saveRe = MySqlUtil.executSql(sql)

        oriStatus['last_status'] = result['last_status']
        oriStatus['period_end'] = result['period']['end'].strftime('%Y-%m-%d %H:%M:%S:%f')
        oriStatus['post_time'] = result['post_time'].strftime('%Y-%m-%d %H:%M:%S:%f')
        if saveRe['lastrowid']!=0:
            oriStatus['result_id'] = saveRe['lastrowid']
        oriStatus['is_reset'] = isReset
        RedisUtil.RedisUtil.setStatus(key, oriStatus)

    #保存event log

    if oriStatus['result_id']!='0':
        eventVal = []
        for event in result['event_list']:
            if event['status ID'] == 0:
                continue
            tup = {}
            tup['_id'] = event['time'] + "_%s" % oriStatus['result_id']
            tup['date'] = nowTime.strftime("%Y-%m-%d")
            tup['lift_id'] = liftId
            tup['result_id'] = "%s" % oriStatus['result_id']
            tup['event_desc'] = "%s" % event['description']
            tup['status_id'] = "%s" % event['status ID']
            tup['time'] = "%s" % event['time']
            eventVal.append(tup)

        if (len(eventVal) > 0):
            MongoUtil.MongodbModule.saveData("db_xgdt", "eventlog_" + nowTime.strftime("%Y%m"), eventVal)







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
    for i in range(10*60 * num + 5, 5, -1):
        tempTime = dateStr + timedelta(seconds=-i)
        tempTimeStr = tempTime.strftime("%Y%m%d%H%M%S")
        for j in range(0,1000,50):
            millisecond='%03d'%j
            times.append(tempTimeStr+millisecond)
    return times



if __name__ =='__main__':
    # %% Define class
    #%% Define classes for Brake, Motor and Door signals, respectively

    #calModeData('2021-08-22 16:42:00')  #2021-06-02 00:00
    t = time.perf_counter()
    dateStr = '2022-12-12 03:00:00'
    # dateList=[]
    for i in range(1, len(sys.argv)):

        dateStr += sys.argv[i]
        # dateList.append(sys.argv[i])

    #if len(sys.argv) >=1:
    #    dateStr = sys.argv[1:]
    print(dateStr)
    hisDate = datetime.strptime(dateStr, '%Y-%m-%d %H:%M:%S')
    for i in range(1, 2):
        currDate = hisDate + timedelta(hours=i)
        currDateStr = currDate.strftime("%Y-%m-%d %H:%M:%S")
        print(currDateStr)
        calModeData(currDateStr)




    print(f'coast:{time.perf_counter() - t:.8f}s')



