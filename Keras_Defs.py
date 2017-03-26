# coding: utf-8

import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks
from keras.layers import Dense,Activation,Dropout,merge
from keras.models import Sequential
from keras.layers import LSTM
from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Nadam,Adamax
from mpl_toolkits.mplot3d import Axes3D
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import random
import math
from keras.models import load_model
from keras.layers import Input
from keras.models import Model
import plotly.graph_objs as go
import plotly.offline as py
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

class showEpoch(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch%10==0:
            print epoch

def Modelling(data,trainstart,trainsamplenum,inputpara,outputpara,dropoutper):

    inputnum= len(inputpara)
    outputnum = len(outputpara)
    input = data[trainstart:trainstart+trainsamplenum, inputpara]
    output = data[trainstart:trainstart+trainsamplenum, outputpara]
    train = np.zeros((trainsamplenum, inputnum))
    lable = np.zeros((trainsamplenum, outputnum))
    Min_input=np.zeros((inputnum))
    Max_input = np.zeros((inputnum))
    Min_output = np.zeros((outputnum))
    Max_output = np.zeros((outputnum))
    for j in range(inputnum):
        Min_input[j]=np.min(input[:,j])
        Max_input[j]=np.max(input[:,j])
    for j in range(outputnum):
        Min_output[j]=np.min(output[:,j])
        Max_output[j]=np.max(output[:,j])
    for i in range(trainsamplenum):
        for j in range(inputnum):
            train[i,j]=(input[i,j]-Min_input[j])/(Max_input[j]-Min_input[j])
        for j in range(outputnum):
            lable[i,j]=(output[i,j]-Min_output[j])/(Max_output[j]-Min_output[j])

    print "-----------------"
    print "modelling"


    # 模型块
    model=Sequential()
    model.add(Dense(32, init='uniform', input_dim=inputnum))
    model.add(Activation('relu'))
    model.add(Dense(32))          #3
    model.add(Activation('relu'))
    model.add(Dense(32))          #4
    model.add(Activation('relu'))
    model.add(Dense(32))          #5
    # model.add(Activation('relu'))
    # model.add(Dense(32))          #6
    # model.add(Activation('relu'))
    # model.add(Dense(32))          #7
    # model.add(Activation('relu'))
    # model.add(Dense(32))          #8
    # model.add(Activation('relu'))

    model.add(Dropout(dropoutper))         #1
    # model.add(Activation('relu'))
    model.add(Dense(outputnum))     #2
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
    # 模型块

    checkpointer = showEpoch()
    print "fitting"
    print "-----------------"
    starttime = datetime.datetime.now()

    # 训练块
    hist = model.fit(train, lable, batch_size=1000, nb_epoch=100, shuffle=True, verbose=0, validation_split=0.0,callbacks=[checkpointer])
    # 训练块

    endtime = datetime.datetime.now()
    print "-----------------"
    print "fit done"
    print "Time consumption:",(endtime - starttime).seconds," s"
    print "-----------------"
    print(hist.history)
    print "-----------------"
    print(hist.history.get('loss')[-1])
    # loss_metrics = model.evaluate(train, lable, batch_size=1)
    print "-----------------"

    return hist,model,input,output,train,lable

def print3D(model,input,output,x3D,y3D,z3D):
    print "3D fig"
    # predict with normalized data
    inputnum=len(input[0,:])
    outputnum = len(output[0, :])
    X=np.arange(0,1,0.05)
    Y=np.arange(0,1,0.05)
    X,Y=np.meshgrid(X,Y)
    Oriarr=np.zeros((len(X)*len(Y),2))
    arr = 0.5 * np.ones((len(X) * len(Y), inputnum))
    m=0
    for i in range (len(X)):
        for j in range(len(Y)):
            Oriarr[m] = np.hstack((X[i][j], Y[i][j]))
            m+=1

    arr[:,x3D]=Oriarr[:,0]
    arr[:,y3D]=Oriarr[:,1]
    result = model.predict(arr, batch_size=1)
    Z=np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[i][j] = result[i*len(Y)+j,z3D-inputnum]

    # Anti normalization
    x=np.zeros((len(X),len(Y)))
    y=np.zeros((len(X),len(Y)))
    z=np.zeros((len(X),len(Y)))

    Min_input_x3D = np.min(input[:, x3D])
    Max_input_x3D = np.max(input[:, x3D])

    Min_input_y3D = np.min(input[:, y3D])
    Max_input_y3D = np.max(input[:, y3D])

    Min_input_z3D_inputnum = np.min(output[:,z3D-inputnum])
    Max_input_z3D_inputnum = np.max(output[:,z3D-inputnum])

    for i in range(len(X)):
        for j in range(len(Y)):
            x[i,j] = X[i, j] * (Max_input_x3D - Min_input_x3D) + Min_input_x3D

            y[i,j] = Y[i, j] * (Max_input_y3D - Min_input_y3D) + Min_input_y3D

            z[i,j]=Z[i,j]*(Max_input_z3D_inputnum-Min_input_z3D_inputnum)+Min_input_z3D_inputnum

    fig=plt.figure()
    ax=Axes3D(fig)
    ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='rainbow')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.title("3D face")
    # plt.show()
    # np.savetxt('resultx.csv', x, delimiter = ',')
    # np.savetxt('resulty.csv', y, delimiter = ',')
    # np.savetxt('resultz.csv', z, delimiter = ',')
    return plt

def preDict(model,data,input,output,inputpara,outputpara,predictstart,predictnum):
    print("predict")
    inputnum = len(input[0, :])
    outputnum = len(output[0, :])
    predictend=predictstart + predictnum
    # 真实数据作为测试输入
    testinput = data[predictstart:predictend, inputpara]
    # 真实数据作为输出，用来对比效果
    testoutput = data[predictstart:predictend, outputpara]
    # 归一化
    test = np.zeros((predictnum, inputnum))

    Min_input = np.zeros((inputnum))
    Max_input = np.zeros((inputnum))
    for j in range(inputnum):
        Min_input[j] = np.min(input[:, j])
        Max_input[j] = np.max(input[:, j])

    for i in range(predictnum):
        for j in range(inputnum):
            test[i, j] = (testinput[i, j] - Min_input[j]) / (Max_input[j] - Min_input[j])

    # 得到模型输出(归一化的)
    testresult = model.predict(test, batch_size=1)

    # 将模型输出进行反归一化
    testresultReal = np.zeros((predictnum, outputnum))

    Min_output = np.zeros((outputnum))
    Max_output = np.zeros((outputnum))
    for j in range(outputnum):
        Min_output[j] = np.min(output[:, j])
        Max_output[j] = np.max(output[:, j])

    for i in range(predictnum):
        for j in range(outputnum):
            testresultReal[i, j] = testresult[i, j] * (Max_output[j] - Min_output[j]) + Min_output[j]
    errAb=np.zeros((predictnum, outputnum))
    errRe = np.zeros((predictnum, outputnum))
    for i in range(outputnum):
        errAb[:,i] = testresultReal[:, i] - testoutput[:, i]  # 模型输出-实际输出
        errRe[:,i] = errAb[:,i] / (Max_output[i] - Min_output[i])  # 相对预测误差
    # for i in range(outputnum):
    #     xaxis=range(preNum)
    #     plt.figure(i)  # 创建图表
    #     ax1 = plt.subplot(221)  # 在图表中创建子图1
    #     ax2 = plt.subplot(222)  # 在图表中创建子图2
    #     ax3 = plt.subplot(223)  # 在图表中创建子图3
    #     ax4 = plt.subplot(224)  # 在图表中创建子图4
    #     plt.sca(ax1)  # 选择图表的子图1
    #     plt.plot(xaxis, errAb[:,i])
    #     plt.title("absolute error " + str(i))
    #     plt.sca(ax2)  # 选择图表的子图2
    #     plt.plot(xaxis, errRe[:,i])
    #     plt.title("related error " + str(i))
    #     plt.sca(ax3)  # 选择图表的子图3
    #     plt.plot(xaxis, testresultReal[:, i])
    #     plt.title("model output " + str(i))
    #     plt.sca(ax4)  # 选择图表的子图4
    #     plt.plot(xaxis, testoutput[:, i])
    #     plt.title("real output " + str(i))
    #     plt.show()
    return testoutput,testresultReal,errAb,errRe


def printLoss(hist):
    plt.figure("Loss")
    plt.plot(range((len(hist.history.get('acc')))), hist.history.get('loss'))
    plt.title("Fitting Loss Function")
    return plt



def sendEmail(Mes):

    SMTPserver = 'smtp.163.com'
    sender = 'shaozhuang_pro@163.com'
    password = "shaozhuang163"
    destination="594459776@qq.com"

    message = Mes
    msg = MIMEText(message)

    msg['Subject'] = 'Email from Python'
    msg['From'] = sender
    msg['To'] = destination

    mailserver = smtplib.SMTP(SMTPserver, 25)
    mailserver.login(sender, password)
    mailserver.sendmail(sender, [destination], msg.as_string())
    mailserver.quit()

    print 'send email success'