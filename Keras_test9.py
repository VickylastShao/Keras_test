# coding: utf-8
# 岱海引风机数据
# 重新考虑输入输出量

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.visualize_util import plot
from Keras_Defs import *
from keras.models import load_model

# OriData = np.loadtxt("DTP.csv", dtype=np.str, delimiter=",")
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
#0~315708
ts,tn=0,1000 #训练样本起始点,训练样本数量
NL=5 #神经网络层数

# 未用
# 总煤量信号(16)
# 1送风机入口风温(17) 2送风机入口风温(18)   二次风#1暖风器出口风压(21) 锅炉蒸发量/主蒸汽流量(22)
# 输入
# 负荷(0) 1号空预器氧量(1) 2号空预器氧量(2) 1空气预热器出口烟温A(4) 1空气预热器出口烟温B(5) 1空气预热器出口烟温C(6) 2空气预热器出口烟温A(7)
# 2空气预热器出口烟温B(8) 2空气预热器出口烟温C(9) 引风机进口静压(12) 引风机动叶开度(14) 炉膛负压(20)
# 输出
# 烟囱入口净烟气氧气(3) 1引风机出口烟温(10) 2引风机出口烟温(11)  引风机出口静压(13)  吸收塔入口烟气流量(15) 1号引风机电流(19)


ip=[0,1,2,4,5,6,7,8,9,12,14,20]      #输入参数index
op=[3]   #输出参数index
# 3,10,11,13,15,19
dropoutper=0.3
hist,model,input,output,train,lable=Modelling(data,ts,tn,ip,op,dropoutper)
loss=hist.history.get('loss')
acc=hist.history.get('acc')
model.save('Simple_models/DH_fan_model_'+str(ts)+'to'+str(ts+tn)+'_'+str(NL)+'L_output'+str(op)+'_Dropout'+str(dropoutper)+'.h5')
np.save('loss.npy',loss)
np.save('acc.npy',acc)
np.save('input.npy',input)
np.save('output.npy',output)
np.save('train.npy',train)
np.save('lable.npy',lable)
# sendEmail('Modelling Done')

model=load_model('Simple_models/DH_fan_model_'+str(ts)+'to'+str(ts+tn)+'_'+str(NL)+'L_output'+str(op)+'_Dropout'+str(dropoutper)+'.h5')
loss=np.load('loss.npy')
acc=np.load('acc.npy')
input=np.load('input.npy')
output=np.load('output.npy')
train=np.load('train.npy')
lable=np.load('lable.npy')

plot(model, to_file='model_'+str(NL)+'L.png')

rangeNum=3000 #30000(start——end)组算例
avgerr=np.zeros((rangeNum,len(op)))
for i in range(rangeNum):
    preStart=i*100
    preNum=100
    testoutput,testresultReal,errAb,errRe=preDict(model,data,input,output,ip,op,preStart,preNum)
    print i
    for j in range(len(op)):
        avgerr[i,j]=np.mean(errRe[:,j])
# xaxis=range(len(op))
# plt.figure("errRe")
# plt.plot(xaxis, avgerr)
# plt.title("related error")
# plt.show()

np.savetxt('Simple_models/avgerr_model_'+str(ts)+'_'+str(ts+tn)+'_pre_start_end_'+str(NL)+'L_output'+str(op)+'_Dropout'+str(dropoutper)+'.csv', avgerr, delimiter = ',')
# sendEmail('preDict Done')

