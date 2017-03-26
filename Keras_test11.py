# coding: utf-8
# 岱海引风机数据
# 自编码机

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.visualize_util import plot
from Keras_Defs import *
from keras.models import load_model
from keras.layers import Input, Dense
from keras.models import Model
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
# np.save('data_filter.npy',data)

data=np.load('data_filter.npy')

#0~153035
ts,tn=0,150000 # trainstart,trainnumber

# 输入 自编码机输入输出均为这些参数
# 负荷(0) 1号空预器氧量(1) 2号空预器氧量(2) 1空气预热器出口烟温A(4) 1空气预热器出口烟温B(5) 1空气预热器出口烟温C(6) 2空气预热器出口烟温A(7)
# 2空气预热器出口烟温B(8) 2空气预热器出口烟温C(9) 引风机进口静压(12) 引风机动叶开度(14) 炉膛负压(20)
# 输出 不参与自编码机建模
# 烟囱入口净烟气氧气(3) 1引风机出口烟温(10) 2引风机出口烟温(11)  引风机出口静压(13)  吸收塔入口烟气流量(15) 1号引风机电流(19)
# # 未用  考虑到这些信号对引风机的影响延迟时间太长，自编码机建模效果不会很好
# 总煤量信号(16)
# 1送风机入口风温(17) 2送风机入口风温(18)   二次风#1暖风器出口风压(21) 锅炉蒸发量/主蒸汽流量(22)
# 训练数据整合
ip=[0,1,2,4,5,6,7,8,9,12,14,20]
op=[0,1,2,4,5,6,7,8,9,12,14,20]
inputnum= len(ip)
input = data[ts:ts+tn, ip]
train = np.zeros((tn, inputnum))
lable = np.zeros((tn, inputnum))
Min_input = np.zeros((inputnum))
Max_input = np.zeros((inputnum))

for j in range(inputnum):
    Min_input[j]=np.min(input[:,j])
    Max_input[j]=np.max(input[:,j])

for i in range(tn):
    for j in range(inputnum):
        train[i,j]=(input[i,j]-Min_input[j])/(Max_input[j]-Min_input[j])
lable=train
# 训练数据整合


print "-----------------"
print "modelling"

# 训练模型块
for mn in range(11-1):
    midnum = 11-mn
    for cn in range(midnum-1):
        codenum=midnum-cn-1

        # Main_Input=Input(shape=(inputnum,),name='Main_Input')
        # encoded1 = Dense(midnum, activation='relu',name='encoded1')(Main_Input)
        # encoded2 = Dense(codenum, activation='relu' ,name='encoded2')(encoded1)
        #
        # decoded1 = Dense(midnum, activation='relu' ,name='decoded1')(encoded2)
        # decoded2 = Dense(inputnum, activation='sigmoid' ,name='decoded2')(decoded1)
        #
        # # 自编码机
        # autoencoder = Model(input=Main_Input, output=decoded2)
        # # 编码机
        # encoder = Model(input=Main_Input, output=encoded2)
        # # 解码机
        # encoded_input = Input(shape=(codenum,),name='decoder_Input')
        # decoder_layer1 = autoencoder.layers[-2](encoded_input)
        # decoder_layer2 = autoencoder.layers[-1](decoder_layer1)
        # decoder = Model(input=encoded_input, output=decoder_layer2)


        # sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        # autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
        #
        # # checkpointer = showEpoch()
        # print "fitting"
        # print "-----------------"
        # starttime = datetime.datetime.now()
        #
        # # 训练块
        # hist=autoencoder.fit(train, lable, batch_size=50000, nb_epoch=1000,
        #                 shuffle=True,validation_split=0.2)
        # # 训练块 , callbacks=[checkpointer]
        # endtime = datetime.datetime.now()
        # print "-----------------"
        # print "fit done"
        # print "Time consumption:",(endtime - starttime).seconds," s"
        # print "-----------------"
        # print(hist.history)
        # print "-----------------"
        # print(hist.history.get('loss')[-1])
        # print "-----------------"
        #
        # # plot(autoencoder, to_file='AutoEncoder/autoencoder.png')
        # # plot(encoder, to_file='AutoEncoder/encoder.png')
        # # plot(decoder, to_file='AutoEncoder/decoder.png')
        #
        # loss=hist.history.get('loss')
        # val_loss=hist.history.get('val_loss')
        # autoencoder.save('AutoEncoder/autoencoder'+str(midnum)+'_code'+str(codenum)+'.h5')
        # # encoder.save('encoder.h5')
        # # decoder.save('decoder.h5')
        #
        # # plt.figure("loss")
        # # plt.plot(loss)
        # # plt.show()
        # #
        # # plt.figure("val_loss")
        # # plt.plot(val_loss)
        # # plt.show()
        #
        # # sendEmail('Modelling Done')
        # # 训练模型块


        # 模型预测
        autoencoder=load_model('AutoEncoder/autoencoder'+str(midnum)+'_code'+str(codenum)+'.h5')
        # encoder=load_model('encoder.h5')
        # decoder=load_model('decoder.h5')
        # 得到模型输出(归一化的)
        testresult = autoencoder.predict(train, batch_size=50000)
        # code=encoder.predict(train,batch_size=50000)
        # result=decoder.predict(code,batch_size=50000)

        # 将模型输出进行反归一化
        testresultReal = np.zeros((tn, inputnum))
        for i in range(tn):
            for j in range(inputnum):
                testresultReal[i, j] = testresult[i, j] * (Max_input[j] - Min_input[j]) + Min_input[j]
        # 将模型输出进行反归一化
        errAb = np.zeros((tn, inputnum))
        errRe = np.zeros((tn, inputnum))
        for i in range(inputnum):
            errAb[:, i] = testresultReal[:, i] - input[:, i]  # 模型输出-实际输出
            errRe[:, i] = errAb[:, i] / (Max_input[i] - Min_input[i])  # 相对预测误差

        avgER=np.zeros((inputnum))
        for i in range(inputnum):
            avgER[i]=np.mean(np.abs(errRe[:, i]))

        np.savetxt('AutoEncoder/AutoEncoder_mid'+str(midnum)+'_code'+str(codenum)+'.csv', avgER, delimiter = ',')

        # plt.figure("errRe")
        # plt.plot(errRe)
        # plt.show()

        # plt.figure("code0")
        # plt.plot(code[:,0])
        # plt.show()
        # plt.figure("code1")
        # plt.plot(code[:,1])
        # plt.show()
        #
        # plt.figure("testresult")
        # plt.plot(testresult)
        # plt.show()
        # plt.figure("result")
        # plt.plot(result)
        # plt.show()


        # plt.figure("scatter x-y-z")
        # ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程
        # ss=5000
        # se=10000
        # ax.scatter(code[ss:se,0], code[ss:se,1],code[ss:se,2]) #绘制数据点
        # plt.show()


        # plt.figure("partsfigure")
        # plt.plot(testresultReal[0:100,0],label='coded output')
        # plt.plot(input[0:100,0],label='input')
        # plt.show()
        # sendEmail('Predict Done')
        # 模型预测

sendEmail('Predict Done')

