import ConstantUtil
import pymysql

def getData(sql):
    if len(ConstantUtil.ConstantModule.dbIP) == 0: ConstantUtil.ConstantModule.loadConfig()  # .loadconnection()
    print(ConstantUtil.ConstantModule.dbIP)
    print(ConstantUtil.ConstantModule.dbUser)
    print(ConstantUtil.ConstantModule.dbPassword)
    print(ConstantUtil.ConstantModule.dbDatabase)

    conn = pymysql.connect(host=ConstantUtil.ConstantModule.dbIP,  port=3306, \
                            user=ConstantUtil.ConstantModule.dbUser,    \
                           password=ConstantUtil.ConstantModule.dbPassword,  \
                          database=ConstantUtil.ConstantModule.dbDatabase,charset="utf8")

    #conn = pymysql.connect(host="47.105.56.147",port=3306,user="root",password="rhxd@147",database="db_xg",charset="utf8")


    # v_center_model_config ia_center_model_config

    cur = conn.cursor()
    cur.execute(sql)
    list = cur.fetchall()
    if len(list) == 0:
        return None
    data=list
    # for col in list:
    #    print(col[0])



    cur.close()
    conn.close()
    return data


def saveData(sql):
    if len(ConstantUtil.ConstantModule.dbIP) == 0: ConstantUtil.ConstantModule.loadConfig()  # .loadconnection()
    print(ConstantUtil.ConstantModule.dbIP)
    print(ConstantUtil.ConstantModule.dbUser)
    print(ConstantUtil.ConstantModule.dbPassword)
    print(ConstantUtil.ConstantModule.dbDatabase)

    conn = pymysql.connect(host=ConstantUtil.ConstantModule.dbIP,  port=3306, \
                            user=ConstantUtil.ConstantModule.dbUser,    \
                           password=ConstantUtil.ConstantModule.dbPassword,  \
                          database=ConstantUtil.ConstantModule.dbDatabase,charset="utf8")

    #conn = pymysql.connect(host="47.105.56.147",port=3306,user="root",password="rhxd@147",database="db_xg",charset="utf8")


    # v_center_model_config ia_center_model_config

    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    data={}
    data['rowcount'] = cur.rowcount
    data['lastrowid'] = cur.lastrowid
    cur.close()
    conn.close()
    return data

def saveBatchData(sql,val):
    if len(ConstantUtil.ConstantModule.dbIP) == 0: ConstantUtil.ConstantModule.loadConfig()  # .loadconnection()
    print(ConstantUtil.ConstantModule.dbIP)
    print(ConstantUtil.ConstantModule.dbUser)
    print(ConstantUtil.ConstantModule.dbPassword)
    print(ConstantUtil.ConstantModule.dbDatabase)

    conn = pymysql.connect(host=ConstantUtil.ConstantModule.dbIP,  port=3306, \
                            user=ConstantUtil.ConstantModule.dbUser,    \
                           password=ConstantUtil.ConstantModule.dbPassword,  \
                          database=ConstantUtil.ConstantModule.dbDatabase,charset="utf8")

    #conn = pymysql.connect(host="47.105.56.147",port=3306,user="root",password="rhxd@147",database="db_xg",charset="utf8")


    # v_center_model_config ia_center_model_config

    cur = conn.cursor()
    cur.executemany(sql,val)
    conn.commit()
    data={}
    data['rowcount']=cur.rowcount
    data['lastrowid']=cur.lastrowid
    cur.close()
    conn.close()
    return data