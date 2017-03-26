# coding: utf-8
# 别人做的三个神经网络建模函数

from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
import keras
from keras.optimizers import SGD
from keras.optimizers import Adam

def DNN(d,m,act,drp,h1):
    model=Sequential()
    model.add(Dense(m,imput_dim=d,activation=act))
    model.add(Dropout(drp))
    for i in range(h1-1):
        model.add(Dense(m,activation=act))
        model.add(Dropout(drp))
    model.add(Dense(2))
    model.add(Activation("linear"))
    #sgd=SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
    adam=keras.optimizers.adam(lr=1e-5,bera_1=0.9,beta_2=0.99,epsilon=1e-8)
    #sgd=SGD(lr=0.01,momentum=0.9,nesterov=True)
    model.compile(adam,'mse')
    return model

def MLP(d,m,q):
    model=Sequential()
    model.add(Dense(m,input_dim=d,activation='sigmoid'))
    model.add(Dense(q))
    model.add(Activation("softmax"))
    model.compile('rmsprop','categorical_crossentropy')
    return model

def PTRN(d,q):
    model=Sequential()
    model.add(Dense(q,input_dim=d))
    model.add(Activation("softmax"))
    model.compile('rmsprop','categorical_crossentropy')
    return model
