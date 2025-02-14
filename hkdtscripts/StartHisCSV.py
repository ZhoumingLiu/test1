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



import glob
import sys
from tqdm import tqdm


        


def  calModeData(csvFile):
    csvFileName = os.path.basename(csvFile)
    csvFileName=csvFileName.split('.')[0]
    print(csvFileName)
    model = csvFileName.split('_')
    liftName= model[0]
    calDate=model[1]
    print(calDate)
    print(liftName)
    sql = "SELECT d.`C0003_DEVID`,d.`C0003_DEVNAME`,d.`C0002_STATION_NO`,d.`C0002_STATION_NAME`,d.`c0003_field31`  FROM sys_m_device d WHERE c0003_devtype='dt' and c0003_field31 is not null  AND clean_flag IS NOT NULL AND clean_flag!=4 and d.C0003_DEVNAME='"+liftName+"'"
    datas = MySqlUtil.getData(sql)
    print(datas)
    
    if(len(datas)==0 or len(datas)>1) :
        return


    MongoUtil.MongodbModule.Init()
    RedisUtil.RedisUtil.Init()

    dataset_raw = pd.read_csv(csvFile,index_col = False)
    # print(dataset_raw)
    dataset = dataset_raw.copy()
    dataset['Time'] = pd.to_datetime(dataset['Time'],errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
    # dataset['Time'] = pd.to_datetime(dataset['Time'],format='%Y-%m-%d %H:%M:%S:%f')
    # dataset['Time'] = dataset['Time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    dataset = dataset.astype({'Motor':'float','Brake':'float','Safety':'float','Door':'float','Resv-1':'float','Resv-2':'float','Resv-3':'float','Distance':'float'})
    
    # 组装为pandas 数据对象
    # df = pd.DataFrame(frameData)
    # print(df)
    # 数据清洗，清洗掉nan数据
    # newdf =df.dropna(subset=['Brake', 'Door'], inplace=True)
    # if newdf is None :
    #    continue
    # 调用柴博士开发好的算法
    #print(data[4])
    module = importlib.import_module('devpy' + datas[0][4])
    operation_class = getattr(module, "runMethod")
    
    #### 每次执行一小时的数据。一共执行24次每天 #####
    
    
    # result,CarSeg_Stats,DoorSeg_Stats = operation_class(df)

    i = 0
    while i<len(dataset):
        # print(i)

        seq = dataset.iloc[i : (i+72000)]
        
        
        # Run the main code
        # Result,_,_ = run_21_EMSDHQL12.runMethod(seq) 
        result,CarSeg_Stats,DoorSeg_Stats = operation_class(seq)        
        
        print(result)
        # txt_file = open("C:/Users/chais/Dropbox/Spyder_Dropbox/Elevator EMSD code/DataFiles/EMSD HQ L12_2022-12.txt", "a", encoding="utf-8")  # 以写的格式打开先打开文件
        # txt_file.write(str(result))
        # txt_file.write("\n")
        # txt_file.close()
        
        #### 对结果进行解析存储 #####

        saveResult(calDate,datas[0][2],datas[0][0],result)
        
        i = i+72000




    
    # return df

def saveResult(nowTime,floorId,liftId,result):
    nowTime = datetime.strptime(nowTime,'%Y-%m-%d')
    #保存result
    isReset=0
    if result['last_status'] == 0:
        isReset=1
    key='his_status:%s'%liftId
    print(key)
    RedisUtil.RedisUtil.Init()
    
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










if __name__ =='__main__':
    # %% Define class
    #%% Define classes for Brake, Motor and Door signals, respectively
    t = time.perf_counter()


    # csvFile ='./EMSD HQ L12_2022-12-11.csv'
    # calModeData(csvFile)


    # print(f'coast:{time.perf_counter() - t:.8f}s')


# 加载该电梯的关键参数
# paras = paras_EMSDHQL12


    directoryPath = "G:/BaiduNetdiskDownload/EMSD/EMSDHQ_L12/202212/"
    
    progress=0 # progress flag
    nfiles = len(glob.glob(directoryPath+'*.csv'))
    # Trip_list = []
    # SpeedSpike_list = []
    # SpikeTime_list = []
    
    for file_name in tqdm(glob.glob(directoryPath+'*.csv')):
        
        calModeData(file_name)
        print(f'coast:{time.perf_counter() - t:.8f}s')
