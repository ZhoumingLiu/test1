import redis
import ConstantUtil

class RedisUtil(object):

    @staticmethod
    def Init():

        RedisUtil.pool = redis.ConnectionPool(host=ConstantUtil.ConstantModule.redisIP, port=ConstantUtil.ConstantModule.redisPort,password=ConstantUtil.ConstantModule.redisPassword , decode_responses=True,db=1)

    def setString(key,value):
        r = redis.Redis(connection_pool=RedisUtil.pool)
        r.set(key,value)

    def getString(key):
        r = redis.Redis(connection_pool=RedisUtil.pool)
        return r.get(key)

    def setStatus( key, value):
        r = redis.Redis(connection_pool=RedisUtil.pool)
        r.hmset(key,{"result_id":value['result_id'],"lift_id":value['lift_id'],"floor_id":value['floor_id'],"last_status":value['last_status'],"period_start":value['period_start'],"period_end":value['period_end'],"post_time":value['post_time'],"is_reset":value['is_reset']})

    def getStatus(key):
        r = redis.Redis(connection_pool=RedisUtil.pool)
        return r.hgetall(key)


