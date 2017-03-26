# coding: utf-8
# 岱海引风机数据

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import datetime
from keras.utils.visualize_util import plot

OriData = np.loadtxt("DT.csv", dtype=np.str, delimiter=",")
data=OriData[1:,:].copy()#去除第一行
data=data[:,1:].copy()#去除第一列

# for i in range(len(data[:,0])):#遍历保证数据都能转为float
#     for j in range(len(data[0,:])):
#         print i,j
#         a=data[i,j].astype(np.float)

data = data.astype(np.float)  # 转类型

trainsamplenum=200  # 训练样本数量

inputnum=7  #前n个参数作为输入
outputnum=len(data[0,:])-inputnum

input=data[:trainsamplenum,:inputnum]# 0~7输入 8~18输出
output=data[:trainsamplenum,inputnum:]
train=np.zeros((trainsamplenum,inputnum))
lable=np.zeros((trainsamplenum,outputnum))

for i in range(trainsamplenum):
    for j in range(inputnum):
        train[i,j]=(input[i,j]-np.min(input[:,j]))/(np.max(input[:,j])-np.min(input[:,j]))
    for j in range(outputnum):
        lable[i,j]=(output[i,j]-np.min(output[:,j]))/(np.max(output[:,j])-np.min(output[:,j]))

print "-----------------"
print "modelling"
model=Sequential()
model.add(Dense(32, init='uniform', input_dim=inputnum))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.3))
# model.add(Activation('relu'))

model.add(Dense(outputnum))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
print "fitting"
starttime = datetime.datetime.now()
hist = model.fit(train, lable, batch_size=10, nb_epoch=200, shuffle=True, verbose=0, validation_split=0.0)
endtime = datetime.datetime.now()
print "Time consumption:",(endtime - starttime).seconds," s"
print "fit done"
plot(model, to_file='model.png')

print(hist.history)
print(hist.history.get('loss')[-1])
loss_metrics = model.evaluate(train, lable, batch_size=1)
plt.plot(range((len(hist.history.get('acc')))),hist.history.get('loss'))
plt.show()
print "-----------------"


model.save('DH_fan_model.h5')
np.save('data.npy',data)
np.save('input.npy',input)
np.save('output.npy',output)
np.save('train.npy',train)
np.save('lable.npy',lable)
np.save('trainsamplenum.npy',trainsamplenum)
np.save('inputnum.npy',inputnum)
np.save('outputnum.npy',outputnum)

