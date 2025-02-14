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

# from utils_lift import *


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
    sql="SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31` ,MOD(d.`C0003_DEVID`,"+hashn+") modv FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null  AND clean_flag IS NOT NULL AND clean_flag!=4 "
    datas = MySqlUtil.getData(sql)
    #meansObj = json.loads(MEANS_JSON)
    #处理单个电梯设备
    MongoUtil.MongodbModule.Init()
    RedisUtil.RedisUtil.Init()
    print(hashn)
    print(currthash)
    threads = []
    for data in datas :
        if (data[4] is  None or len(data[4]) == 0):
            continue
        #分布式执行判断语句
        print(data)
        if(data[5]!=int(currthash)):
             continue
        print('start')
        #print('thread')
        thread=threading.Thread(target=devThread,args=(data,times,nowTime))
        thread.start()
        threads.append(thread)
        thread.join()
        


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
    #print(data[4])
    module = importlib.import_module('devpy' + data[4])
    operation_class = getattr(module, "runMethod")
    result,CarSeg_Stats,DoorSeg_Stats = operation_class(df)
    # result = run_4008636142EMSDHQL6.runMethod(newdf)
    print(result)
    # print(CarSeg_Stats)
    # 对结果进行解析存储
    saveResult(nowTime,data[2],data[0],result,CarSeg_Stats,DoorSeg_Stats)
    #print(DoorSeg_Stats)

    
    # return df

def saveResult(nowTime,floorId,liftId,result,CarSeg_Stats,DoorSeg_Stats):
    #保存result
    isReset=0
    if result['last_status'] == 0:
        isReset=1
    key='status:%s'%liftId
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
        tempList=[]
        for event in result['event_list']:
            if event['status ID'] == 0 or event['time'] =='':
                continue
            eventTime = datetime.strptime(event['time'],'%Y-%m-%d %H:%M:%S.%f')
            tup = {}
            tup['_id'] = event['time'] + "_%s" % oriStatus['result_id']
            tup['date'] = eventTime.strftime("%Y-%m-%d")
            tup['lift_id'] = liftId
            tup['result_id'] = "%s" % oriStatus['result_id']
            tup['event_desc'] = "%s" % event['description']
            tup['status_id'] = "%s" % event['status ID']
            tup['time'] = "%s" % event['time']
            tup['floor'] = "%s" % event['floor']
            tup['delsign'] = "%s" % event['delsign']
            if tup['_id'] not in tempList:
                eventVal.append(tup)
                tempList.append(tup['_id'])
            else:
                for tempTup in eventVal:
                    if tempTup['_id']==tup['_id']:
                        tempTup['event_desc'] = tempTup['event_desc']+";%s"% event['description']




        if (len(eventVal) > 0):
            MongoUtil.MongodbModule.saveData("db_xgdt", "eventlog_" + eventTime.strftime("%Y%m"), eventVal)




    #保存 CarSeg_Stats
    carstat = []

    for carseg in CarSeg_Stats['CarStat_list']:
        tup={}

        carsegTime = datetime.strptime(carseg['start_time'], '%Y-%m-%d %H:%M:%S.%f')

        tup['_id']= carseg['end_time']+"_"+liftId
        tup['date'] = carsegTime.strftime("%Y-%m-%d")
        tup['lift_id'] = liftId
        tup['start_time'] = "%s"%carseg['start_time']
        tup['end_time'] = "%s"%carseg['end_time']
        tup['duration'] ="%s"% carseg['duration']
        tup['hour'] = "%s"%carseg['hour']
        tup['mileage'] = "%s"%carseg['mileage']
        tup['Depart_F'] = "%s"%carseg['Depart_F']
        tup['Arrive_F'] ="%s"% carseg['Arrive_F']
        tup['Fs_travelled'] ="%s"% carseg['Fs_travelled']
        tup['F_Travelled'] = "%s"%carseg['F_Travelled']
        tup['Dir_Travelled'] = "%s"%carseg['Dir_Travelled']
        tup['MotorI_start'] = "%s"%carseg['MotorI_start']
        tup['MotorI_peak'] = "%s"%carseg['MotorI_peak']
        tup['BrakeI_steady'] = "%s"%carseg['BrakeI_steady']
        tup['BrakeI_peak'] = "%s"%carseg['BrakeI_peak']
        tup['SafetyI_run'] = "%s"%carseg['SafetyI_run']
        tup['Resv1I_start'] = "%s"%carseg['Resv1I_start']
        tup['Resv1I_peak'] = "%s"%carseg['Resv1I_peak']
        tup['Resv2I_start'] = "%s"%carseg['Resv2I_start']
        tup['Resv2I_peak'] = "%s"%carseg['Resv2I_peak']
        tup['Resv3I_run'] = "%s"%carseg['Resv3I_run']
        tup['Speed_peak'] = "%s"%carseg['Speed_peak']

        tup['MotorI_brake'] = "%s" % carseg['MotorI_brake']
        tup['MotorI_steady'] = "%s" % carseg['MotorI_steady']
        tup['Resv1I_brake'] = "%s" % carseg['Resv1I_brake']
        tup['Resv1I_steady'] = "%s" % carseg['Resv1I_steady']
        tup['Resv2I_brake'] = "%s" % carseg['Resv2I_brake']
        tup['Resv2I_steady'] = "%s" % carseg['Resv2I_steady']

        carstat.append(tup);
    if(len(carstat)>0):
        MongoUtil.MongodbModule.saveData("db_xgdt", "carsegstats_" + carsegTime.strftime("%Y%m"), carstat)





    #保存 DoorSeg_Stats

    doorstat = []
    for doorseg in DoorSeg_Stats['DoorStat_list']:
        if(doorseg['duration']<=59):

            tup = {}
            doorsegTime = datetime.strptime(doorseg['start_time'], '%Y-%m-%d %H:%M:%S.%f')
            tup['_id'] = doorseg['end_time'] + "_" + liftId
            tup['date'] = doorsegTime.strftime("%Y-%m-%d")
            tup['lift_id'] = liftId
            tup['start_time'] = doorseg['start_time']
            tup['end_time'] = doorseg['end_time']
            tup['duration'] = "%s"%doorseg['duration']
            tup['hour'] = "%s"%doorseg['hour']
            tup['DoorI_peak'] = "%s"%doorseg['DoorI_peak']
            tup['num_Door'] = "%s"%doorseg['num_Door']
            tup['DoorOpen_Duration'] = "%s"%doorseg['DoorOpen_Duration']
            tup['Stop_F'] = "%s" % doorseg['Stop_F']
            doorstat.append(tup);
    if (len(doorstat) > 0):
        MongoUtil.MongodbModule.saveData("db_xgdt", "doorsegstats_" + doorsegTime.strftime("%Y%m"), doorstat)










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
    dateStr = ''
    # dateList=[]
    for i in range(1, len(sys.argv)):
        print(sys.argv[i])
        dateStr += sys.argv[i];
        # dateList.append(sys.argv[i])
    if len(sys.argv) < 2:
        print('No params specified.')

    if len(sys.argv) >=3:
        hashn = sys.argv[1]
        currthash = sys.argv[2]
    if len(sys.argv) >=4:
        dateStr = sys.argv[3:]

    #calModeData(dateStr)
    calModeData('')
    print(f'coast:{time.perf_counter() - t:.8f}s')



