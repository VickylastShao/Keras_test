# coding: utf-8
# 岱海引风机数据

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.visualize_util import plot
from Keras_Defs import *
from keras.models import load_model

# OriData = np.loadtxt("DT.csv", dtype=np.str, delimiter=",")
# data=OriData[1:,:].copy()#去除第一行
# data=data[:,1:].copy()#去除第一列
#
# # for i in range(len(data[:,0])):#遍历保证数据都能转为float
# #     for j in range(len(data[0,:])):
# #         print i,j
# #         a=data[i,j].astype(np.float)
#
# data = data.astype(np.float)  # 转类型
# np.save('data.npy',data)

data=np.load('data.npy')
#
ts,tn=0,323000 #训练样本起始点,训练样本数量
NL=4 #神经网络层数
ip=[0,1,2,3,4,5,6]      #输入参数index
op=[7,8,9,10,11,12,13,14,15,16,17,18]   #输出参数index

hist,model,input,output,train,lable=Modelling(data,ts,tn,ip,op)

model.save('H5_models/DH_fan_model_'+str(ts)+'to'+str(ts+tn)+'_'+str(NL)+'L.h5')
np.save('hist.npy',hist)
np.save('input.npy',input)
np.save('output.npy',output)
np.save('train.npy',train)
np.save('lable.npy',lable)
sendEmail('Modelling Done')


model=load_model('H5_models/DH_fan_model_'+str(ts)+'to'+str(ts+tn)+'_'+str(NL)+'L.h5')
hist=np.load('hist.npy')
input=np.load('input.npy')
output=np.load('output.npy')
train=np.load('train.npy')
lable=np.load('lable.npy')

plot(model, to_file='model_'+str(NL)+'L.png')

rangeNum=323 #323(start——end) 224(100000——end) 274(50000——end)组算例
avgerr=np.zeros((rangeNum,len(op)))
for i in range(rangeNum):
    preStart=i*1000
    preNum=1000
    testoutput,testresultReal,errAb,errRe=preDict(model,data,input,output,ip,op,preStart,preNum)
    print i
    for j in range(len(op)):
        avgerr[i,j]=np.mean(errRe[:,j])
# xaxis=range(len(op))
# plt.figure("errRe")
# plt.plot(xaxis, avgerr)
# plt.title("related error")
# plt.show()

np.savetxt('avgerr_model_'+str(ts)+'_'+str(ts+tn)+'_pre_start_end_'+str(NL)+'L.csv', avgerr, delimiter = ',')
sendEmail('preDict Done')