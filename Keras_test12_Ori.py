# coding: utf-8
# 这里建立一个时间序列预测模型的demo

# LSTM for international airline passengers problem with regression framing

import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import LSTM
import numpy as np
from keras.optimizers import SGD
from Keras_Defs import *

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
# numpy.random.seed(7)
# load the dataset
# dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[0], engine='python')
# dataset = dataframe.values
# dataset = dataset.astype('float32')
oridataset = np.loadtxt('LSTM/international-airline-passengers.csv', dtype=np.float, delimiter=",")

# normalize the dataset
maxdata=np.max(oridataset)
mindata=np.min(oridataset)
dataset=np.zeros((len(oridataset)))
for i in range(len(oridataset)):
    dataset[i]=(oridataset[i]-mindata)/(maxdata-mindata)
# split into train and test sets


result0=[]
result1=[]
result2=[]

size=0.5

train_size = int(len(dataset) * size)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

look_back=10

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print trainX.shape
print testX.shape

LSTMpointnum=20

# create and fit the LSTM network
model = Sequential()

model.add(LSTM(LSTMpointnum, input_dim=look_back))

model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')
hist=model.fit(trainX, trainY, nb_epoch=100, batch_size=91, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
realtrainPredict=np.zeros((len(trainPredict)))
realtrainY=np.zeros((len(trainPredict)))
for i in range(len(trainPredict)):
    realtrainPredict[i]=trainPredict[i]*(maxdata-mindata)+mindata
    realtrainY[i]=trainY[i]*(maxdata-mindata)+mindata

realtestPredict=np.zeros((len(testPredict)))
realtestY=np.zeros((len(testPredict)))
for i in range(len(testPredict)):
    realtestPredict[i]=testPredict[i]*(maxdata-mindata)+mindata
    realtestY[i]=testY[i]*(maxdata-mindata)+mindata

# calculate root mean squared error
Tempscore=0
for i in range(len(realtrainY)):
    Tempscore=Tempscore+math.pow(realtrainY[i]-realtrainPredict[i],2)
trainScore=math.sqrt(Tempscore/len(realtrainY))

Tempscore=0
for i in range(len(realtestY)):
    Tempscore=Tempscore+math.pow(realtestY[i]-realtestPredict[i],2)
testScore=math.sqrt(Tempscore/len(realtestY))

# plot baseline and predictions
loss=hist.history.get('loss')
plt.figure("loss")
plt.plot(loss)
plt.show()

plt.figure()
plt.plot(range(look_back,len(realtrainPredict)+look_back,1),realtrainPredict[:],'r')
plt.plot(range(len(trainPredict)+(look_back*2)+1,len(dataset)-1,1),realtestPredict[:],'g')

plt.plot(range(0,len(oridataset)),oridataset[0:len(oridataset)],'b')
plt.show()




