# coding: utf-8
# 岱海引风机数据
# 找一些样本数据集，通过调整多层神经网络的结构，对比权值共享与不共享情况下的预测效果
# 备注：小木虫上面已经有过讨论http://muchong.com/html/200912/1723959.html
# 无论网络结构如何，始终采用前200000组数组做训练，后100000组数据做预测

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.visualize_util import plot
from Keras_Defs import *
from keras.models import load_model
from keras.layers import Input, Dense, merge
from keras.models import Model

data=np.load('data.npy')
#0~315708
ts,tn=0,200000 # trainstart,trainnumber

# 未用
# 总煤量信号(16)
# 1送风机入口风温(17) 2送风机入口风温(18)   二次风#1暖风器出口风压(21) 锅炉蒸发量/主蒸汽流量(22)
# 输入
# 负荷(0) 1号空预器氧量(1) 2号空预器氧量(2) 1空气预热器出口烟温A(4) 1空气预热器出口烟温B(5) 1空气预热器出口烟温C(6) 2空气预热器出口烟温A(7)
# 2空气预热器出口烟温B(8) 2空气预热器出口烟温C(9) 引风机进口静压(12) 引风机动叶开度(14) 炉膛负压(20)
# 输出
# 烟囱入口净烟气氧气(3) 1引风机出口烟温(10) 2引风机出口烟温(11)  引风机出口静压(13)  吸收塔入口烟气流量(15) 1号引风机电流(19)
ip=[0,1,2,4,5,6,7,8,9,12,14,20]      # input index
opL=[3,10,11]   # outputLeft index
opR=[13,15,19]  # outputRight index
# 3,10,11,13,15,19
dropoutper=0.3

# 训练数据整合
inputnum= len(ip)
outputnumL = len(opL)
outputnumR = len(opR)
input = data[ts:ts+tn, ip]
outputL = data[ts:ts+tn, opL]
outputR = data[ts:ts+tn, opR]
train = np.zeros((tn, inputnum))
lableL = np.zeros((tn, outputnumL))
lableR = np.zeros((tn, outputnumR))
Min_input=np.zeros((inputnum))
Max_input = np.zeros((inputnum))
Min_outputL = np.zeros((outputnumL))
Max_outputL = np.zeros((outputnumL))
Min_outputR = np.zeros((outputnumR))
Max_outputR = np.zeros((outputnumR))
for j in range(inputnum):
    Min_input[j]=np.min(input[:,j])
    Max_input[j]=np.max(input[:,j])
for j in range(outputnumL):
    Min_outputL[j]=np.min(outputL[:,j])
    Max_outputL[j]=np.max(outputL[:,j])
for j in range(outputnumR):
    Min_outputR[j] = np.min(outputR[:, j])
    Max_outputR[j] = np.max(outputR[:, j])
for i in range(tn):
    for j in range(inputnum):
        train[i,j]=(input[i,j]-Min_input[j])/(Max_input[j]-Min_input[j])
    for j in range(outputnumL):
        lableL[i,j]=(outputL[i,j]-Min_outputL[j])/(Max_outputL[j]-Min_outputL[j])
    for j in range(outputnumR):
        lableR[i,j]=(outputR[i,j]-Min_outputR[j])/(Max_outputR[j]-Min_outputR[j])
# 训练数据整合

print "-----------------"
print "modelling"


# 模型块

Main_Input=Input(shape=(inputnum,),name='Main_Input')
# Main_Dense=Dense(32,activation='relu',name='Main_Dense')(Main_Input)
# Main_Dense2=Dense(32,activation='relu',name='Main_Dense2')(Main_Dense1)
# Main_Dense3=Dense(32,activation='relu',name='Main_Dense3')(Main_Dense2)
# Dropout=Dropout(0.3,name='Dropout')(Main_Dense3)

# Leftpart
DenseL1=Dense(32,activation='relu',name='DenseL1')(Main_Input)
DenseL2=Dense(32,activation='relu',name='DenseL2')(DenseL1)
DenseL3=Dense(32,activation='relu',name='DenseL3')(DenseL2)
DropoutL=Dropout(0.3,name='DropoutL')(DenseL3)
OutputL=Dense(outputnumL,activation='sigmoid',name='OutputL')(DropoutL)

# Rightpart
DenseR1=Dense(32,activation='relu',name='DenseR1')(Main_Input)
DenseR2=Dense(32,activation='relu',name='DenseR2')(DenseR1)
DenseR3=Dense(32,activation='relu',name='DenseR3')(DenseR2)
DropoutR=Dropout(0.3,name='DropoutR')(DenseR3)
OutputR=Dense(outputnumR,activation='sigmoid',name='OutputR')(DropoutR)

# 定义模型
model = Model(input=[Main_Input], output=[OutputL, OutputR])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy',loss_weights=[0.5, 0.5])

# 模型块

checkpointer = showEpoch()
print "fitting"
print "-----------------"
starttime = datetime.datetime.now()

# 训练块
hist = model.fit(train, [lableL, lableR], batch_size=1000, nb_epoch=100, shuffle=True, verbose=0, validation_split=0.3,callbacks=[checkpointer])
# 训练块

endtime = datetime.datetime.now()
print "-----------------"
print "fit done"
print "Time consumption:",(endtime - starttime).seconds," s"
print "-----------------"
print(hist.history)
print "-----------------"
print(hist.history.get('loss')[-1])
print "-----------------"

loss=hist.history.get('loss')
acc=hist.history.get('acc')
model.save('Share_Weight/model_122222.h5')
np.save('Share_Weight/loss_122222.npy',loss)
np.save('Share_Weight/acc_122222.npy',acc)
sendEmail('Modelling Done')

model=load_model('Share_Weight/model_122222.h5')
loss=np.load('Share_Weight/loss_122222.npy')
acc=np.load('Share_Weight/acc_122222.npy')

plot(model, to_file='Share_Weight/TreeModel_122222.png')

rangeNum=300000 #300000(start——end)组算例
avgerr=np.zeros((rangeNum,len(opL)))

preStart=200000
preNum=100000
print("predict")
inputnum = len(input[0, :])
outputnumL = len(outputL[0, :])
outputnumR = len(outputR[0, :])

predictend = preStart + preNum
# 真实数据作为测试输入
testinput = data[preStart:predictend, ip]
# 真实数据作为输出，用来对比效果
testoutputL = data[preStart:predictend, opL]
testoutputR = data[preStart:predictend, opR]
# 归一化
test = np.zeros((preNum, inputnum))

Min_input = np.zeros((inputnum))
Max_input = np.zeros((inputnum))
for j in range(inputnum):
    Min_input[j] = np.min(input[:, j])
    Max_input[j] = np.max(input[:, j])

for i in range(preNum):
    for j in range(inputnum):
        test[i, j] = (testinput[i, j] - Min_input[j]) / (Max_input[j] - Min_input[j])

# 得到模型输出(归一化的)
result = model.predict(test, batch_size=1)
resultL=result[0]
resultR=result[1]
# 将模型输出进行反归一化
resultRealL = np.zeros((preNum, outputnumL))
resultRealR = np.zeros((preNum, outputnumR))

Min_outputL = np.zeros((outputnumL))
Max_outputL = np.zeros((outputnumL))
Min_outputR = np.zeros((outputnumR))
Max_outputR = np.zeros((outputnumR))
for j in range(outputnumL):
    Min_outputL[j] = np.min(outputL[:, j])
    Max_outputL[j] = np.max(outputL[:, j])
for j in range(outputnumR):
    Min_outputR[j] = np.min(outputR[:, j])
    Max_outputR[j] = np.max(outputR[:, j])
for i in range(preNum):
    for j in range(outputnumL):
        resultRealL[i, j] = resultL[i, j] * (Max_outputL[j] - Min_outputL[j]) + Min_outputL[j]
    for j in range(outputnumR):
        resultRealR[i, j] = resultR[i, j] * (Max_outputR[j] - Min_outputR[j]) + Min_outputR[j]
errAbL = np.zeros((preNum, outputnumL))
errReL = np.zeros((preNum, outputnumL))
errAbR = np.zeros((preNum, outputnumR))
errReR = np.zeros((preNum, outputnumR))
for i in range(outputnumL):
    errAbL[:, i] = resultRealL[:, i] - testoutputL[:, i]  # 模型输出-实际输出
    errReL[:, i] = errAbL[:, i] / (Max_outputL[i] - Min_outputL[i])  # 相对预测误差
for i in range(outputnumR):
    errAbR[:, i] = resultRealR[:, i] - testoutputR[:, i]  # 模型输出-实际输出
    errReR[:, i] = errAbR[:, i] / (Max_outputR[i] - Min_outputR[i])  # 相对预测误差


np.savetxt('Share_Weight/resultRealL_122222.csv', resultRealL, delimiter = ',')
np.savetxt('Share_Weight/testoutputL_122222.csv', testoutputL, delimiter = ',')
np.savetxt('Share_Weight/errAbL_122222.csv', errAbL, delimiter = ',')
np.savetxt('Share_Weight/errReL_122222.csv', errReL, delimiter = ',')

np.savetxt('Share_Weight/resultRealR_122222.csv', resultRealR, delimiter = ',')
np.savetxt('Share_Weight/testoutputR_122222.csv', testoutputR, delimiter = ',')
np.savetxt('Share_Weight/errAbR_122222.csv', errAbR, delimiter = ',')
np.savetxt('Share_Weight/errReR_122222.csv', errReR, delimiter = ',')

sendEmail('preDict Done')