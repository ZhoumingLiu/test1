class ConstantModule(object):
    dbIP=''
    dbUser=''
    dbPassword=''
    dbDatabase=''
    mongoIP = ''
    mongoUser = ''
    mongoPassword = ''


    # 静态函数，加载配置文件
    @staticmethod
    def loadConfig():

        import os
        import sys
        dir = os.path.dirname(sys.argv[0])

        congfile = os.path.join(dir, 'config.txt')
        ##读取文本文件获取odbc连接串
        f = open(congfile)
        for lin in f:
            tmp = lin.strip('\n').split('=')
            if tmp[0] == 'db_ip':
                ConstantModule.dbIP = tmp[1]
            elif tmp[0] == 'db_user':
                ConstantModule.dbUser = tmp[1]
            elif tmp[0] == 'db_password':
                ConstantModule.dbPassword = tmp[1]
            elif tmp[0] == 'db_database':
                ConstantModule.dbDatabase = tmp[1]
            elif tmp[0] == 'mongo_ip':
                ConstantModule.mongoIP = tmp[1]
            elif tmp[0] == 'mongo_user':
                ConstantModule.mongoUser = tmp[1]
            elif tmp[0] == 'mongo_password':
                ConstantModule.mongoPassword = tmp[1]



        f.close()


if __name__ =='__main__':
    ConstantModule.loadConfig()
    print(ConstantModule.mongoIP)