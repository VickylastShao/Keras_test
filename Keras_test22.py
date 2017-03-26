# coding: utf-8

import pymysql.cursors
# 在python中实现对MySQL的操作
# 要熟悉参数化编程语句

config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123',
    'db': 'mypro',#数据库名称
    'charset': 'gbk',
    'cursorclass': pymysql.cursors.DictCursor,
}

# connection = pymysql.connect(**config)
# try:
#     with connection.cursor() as cursor:
#         sql = 'INSERT INTO netinfo  VALUES (%s, %s, %s, %s)'
#         cursor.execute(sql, (7, 0.8249, "7","y"))
#     connection.commit()
# finally:
#     connection.close()

TableName='netinfo'
connection = pymysql.connect(**config)
try:
    with connection.cursor() as cursor:
        sql = 'SELECT * FROM '+ TableName
        cursor.execute(sql)
        result = cursor.fetchall()
        results=list(result)
        for r in results:
            # print r
            # 取字典中键neu_value的值,如果不存在该键,则返回0
            print(r.get('neu_value',0))
        connection.commit()
        # sql提交事务,如果没有执行该语句,那上面操作对数据库的修改就是无效的


finally:
    connection.close()



# 常用MySQL语句
# create database MyPro character set utf8
# show databases
# use mypro
# create table netinfo
# (
# 	layer_id int key not null ,
# 	neu_value float not null,
# 	des varchar(25)
# )

# show tables
# insert into netinfo values(1, 0.5926, "1")
# select * from netinfo;
# select neu_value from netinfo
# alter table netinfo change layer_id layer_id int key
# update netinfo set neu_value=0.6012 where layer_id=1
# delete from netinfo where layer_id=2;
# alter table netinfo add tick2 datetime;
# alter table netinfo add tick2 float after neu_value;
# alter table netinfo change tick tickchar char(13) default "-";
# alter table netinfo drop tick2 ;
# alter table netinfo_test rename netinfo;
# drop table netinfo
# drop database mypro
