import pymongo
import ConstantUtil

class MongodbModule(object):



    @staticmethod
    def Init():
        MongodbModule.monClient = pymongo.MongoClient("mongodb://"+ConstantUtil.ConstantModule.mongoIP+"/")
        admindb = MongodbModule.monClient["admin"]
        admindb.authenticate(ConstantUtil.ConstantModule.mongoUser,ConstantUtil.ConstantModule.mongoPassword)

    #查询数据
    def findData(dbName,collName,ids):
        db=MongodbModule.monClient[dbName]
        coll=db[collName]
        datas=coll.find({"_id":{"$in":ids}})
        results={}
        for data in datas:
            results[data['_id'].split('_')[0]]=data
        return results

    #批量保存语句
    def saveData(dbName, collName, dataList):
        keys=[]
        for data in dataList:
            keys.append(data['_id'])
        db = MongodbModule.monClient[dbName]
        coll = db[collName]
        coll.delete_many({"_id":{ "$in"  :keys } })
        rids = coll.insert_many(dataList)
        #print(rids.inserted_ids)

