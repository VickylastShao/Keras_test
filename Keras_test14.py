# coding: utf-8
# 重做加热器LSTM模型
# 注意是循环网络而不是前馈网络

from Keras_Defs import *
import math
lossLIST=[]
trainScoreLIST=[]
testScoreLIST=[]
OP=0
for LF in range(3):
    DataInput=np.load('LSTM_HPH/DataInput.npy')
    DataOutput=np.load('LSTM_HPH/DataOutput.npy')

    maxInput=np.max(DataInput,axis=0)
    minInput=np.min(DataInput,axis=0)
    maxOutput=np.max(DataOutput,axis=0)
    minOutput=np.min(DataOutput,axis=0)

    sizeInput=DataInput.shape
    sizeOutput=DataOutput.shape

    Nordatainput=np.empty_like(DataInput)
    Nordataoutput=np.empty_like(DataOutput)

    # 归一化处理
    # for i in range(sizeInput[0]):
    #     for j in range(sizeInput[1]):
    #         Nordatainput[i,j]=(DataInput[i,j]-minInput[j])/(maxInput[j]-minInput[j])
    #     for j in range(sizeOutput[1]):
    #         Nordataoutput[i,j]=(DataOutput[i,j]-minOutput[j])/(maxOutput[j]-minOutput[j])
    #
    # np.save('LSTM_HPH/Nordatainput.npy',Nordatainput)
    # np.save('LSTM_HPH/Nordataoutput.npy',Nordataoutput)

    Nordatainput=np.load('LSTM_HPH/Nordatainput.npy')
    Nordataoutput=np.load('LSTM_HPH/Nordataoutput.npy')

    trainPer=0.3 #训练样本比例
    train_size= int(sizeInput[0] * trainPer)
    test_size = int(sizeInput[0] - train_size)
    trainin= Nordatainput[0:train_size,:]
    testin = Nordatainput[train_size:sizeInput[0]]
    trainout = Nordataoutput[0:train_size,:]
    testinout = Nordataoutput[train_size:sizeInput[0]]

    #0:蒸汽流量、1:出口给水温度、2:水位、3:出口疏水温度
    outPara=3
    # 0:--++++?  #出口给水温度
    # 1:----++=  #疏水阀门开度
    # 2:---+++=  #蒸汽入口压力
    # 3:-+++++=  #蒸汽入口温度
    # 4:++++++=  #给水流量
    # 5:---+++=  #给水入口温度


    trainX, trainY = trainin,trainout[:,outPara]
    testX, testY = testin,testinout[:,outPara]

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(1024, input_dim=sizeInput[1],return_sequences=False))
    # model.add(LSTM(LSTMpointnum,return_sequences=False))
    # model.add(Dense(512))                #40
    # model.add(Activation('relu'))
    model.add(Dense(512))                #20
    model.add(Activation('relu'))
    # model.add(Dense(128))                #10
    # model.add(Activation('relu'))
    # model.add(Dense(64))                 #5
    # model.add(Activation('relu'))
    model.add(Dense(64))                 #1
    model.add(Activation('relu'))
    # model.add(Dense(16))                 #1
    # model.add(Activation('relu'))
    # model.add(Dense(8))                 #1
    # model.add(Activation('relu'))
    # model.add(Dense(4))                 #1
    # model.add(Activation('relu'))
    # model.add(Dense(2))                 #1
    # model.add(Activation('relu'))
    model.add(Dense(1))                 #1
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rMSprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    adagrad=Adagrad(lr=0.01, epsilon=1e-06)
    adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nadam=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    adamax=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lossF=['squared_hinge',
           'mean_squared_error',
           'binary_crossentropy'
           ]

    if OP==0:
        model.compile(loss=lossF[LF], optimizer=sgd)
    elif OP==1:
        model.compile(loss=lossF[LF], optimizer=rMSprop)
    elif OP==2:
        model.compile(loss=lossF[LF], optimizer=adagrad)
    elif OP==3:
        model.compile(loss=lossF[LF], optimizer=adadelta)
    elif OP==4:
        model.compile(loss=lossF[LF], optimizer=adam)
    elif OP==5:
        model.compile(loss=lossF[LF], optimizer=nadam)
    elif OP==6:
        model.compile(loss=lossF[LF], optimizer=adamax)


    hist=model.fit(trainX, trainY, nb_epoch=500, batch_size=1000, verbose=2, validation_split=0.3)
    # batch_size=3000, verbose=2, validation_split=0.3

    # make predictions
    trainPredict = model.predict(trainX)#训练数据预测结果(0~1)
    testPredict = model.predict(testX)#测试数据预测结果(0~1)
    # invert predictions
    realtrainPredict=np.zeros((len(trainPredict)))
    realtrainY=np.zeros((len(trainPredict)))

    maxdata=maxOutput[outPara]
    mindata=minOutput[outPara]

    for i in range(len(trainPredict)):
        realtrainPredict[i]=trainPredict[i]*(maxdata-mindata)+mindata#训练数据预测结果(正常)
        realtrainY[i]=trainY[i]*(maxdata-mindata)+mindata#训练数据真实结果(正常)

    realtestPredict=np.zeros((len(testPredict)))
    realtestY=np.zeros((len(testPredict)))
    for i in range(len(testPredict)):
        realtestPredict[i]=testPredict[i]*(maxdata-mindata)+mindata#测试数据预测结果(正常)
        realtestY[i]=testY[i]*(maxdata-mindata)+mindata#测试数据真实结果(正常)

    # calculate root mean squared error
    Tempscore=0
    for i in range(len(realtrainY)):
        Tempscore=Tempscore+math.pow(realtrainY[i]-realtrainPredict[i],2)
    trainScore=math.sqrt(Tempscore/len(realtrainY))

    Tempscore=0
    for i in range(len(realtestY)):
        Tempscore=Tempscore+math.pow(realtestY[i]-realtestPredict[i],2)
    testScore=math.sqrt(Tempscore/len(realtestY))

    print('Train Score: %.8f RMSE' % (trainScore))
    print('Test Score: %.8f RMSE' % (testScore))
    # plot baseline and predictions
    loss=hist.history.get('loss')

    lossLIST.append(loss)
    trainScoreLIST.append(trainScore)
    testScoreLIST.append(testScore)

    # plt.figure("loss")
    # plt.plot(loss)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(range(0,len(realtrainY)),realtrainPredict,'r')
    #
    # plt.plot(range(len(realtrainY),sizeOutput[0],1),realtestPredict,'g')
    #
    # plt.plot(range(0,len(DataOutput[:,outPara])),DataOutput[:,outPara],'b')
    # plt.show()

np.save('LSTM_HPH/lossLIST.npy',lossLIST)
np.save('LSTM_HPH/trainScoreLIST.npy',trainScoreLIST)
np.save('LSTM_HPH/testScoreLIST.npy',testScoreLIST)
sendEmail('LSTM Done')