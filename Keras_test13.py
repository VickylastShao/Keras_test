# coding: utf-8
# 这里建立加热器LSTM动态模型
# 数据来自apros仿真结果


from Keras_Defs import *
import math

# 模型输入：
# ValvaPos	InSteamPre	InSteamTem	InWaterMF	InWaterTem
# 疏水阀门开度、蒸汽入口压力、蒸汽入口温度、给水流量、给水入口温度
# 模型输出：
# SteamMF	outSteamTem	WaterLevel	outWaterTem
# 蒸汽流量、出口给水温度、水位、出口疏水温度

# 读取原始数据
# DataInput = np.loadtxt('LSTM_HPH/DLHPH/Model_Input.csv', dtype=np.float, delimiter=",")
# DataOutput = np.loadtxt('LSTM_HPH/DLHPH/Model_Output.csv', dtype=np.float, delimiter=",")
#
# np.save('LSTM_HPH/DataInput.npy',DataInput)
# np.save('LSTM_HPH/DataOutput.npy',DataOutput)

index1=[]
index2=[]
TRS = []
TES = []
D1=[5,16,24,32,40,48,56]
D2=[4,8,12,16,20,24,28]
D3=[3,4,6,8,10,12,14]
D4=[2,2,3,4,5,6,7]
for test1 in range(7):
    for test2 in range(5):
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

        look_backout=50                     #50
        look_backin=[20,30,50,60,30]        #20,30,50,60,30

        look_back = np.max([look_backout, np.max(look_backin)])
        feature = look_backout + np.sum(look_backin) + len(look_backin)

        def create_TimeSeries(datasetins,datasetout, look_backin,look_backout):
            sample = len(datasetout) - look_back
            DataX=np.empty((sample, feature))
            DataY=[]
            for i in range(sample):
                # 预测参数的输入数据
                dataX = []
                outtemp = datasetout[i + (look_back - look_backout):i + (look_back)]
                for j in range(len(outtemp)):
                    dataX.append(outtemp[j])
                # 辅助参数的输入数据
                for j in range(len(look_backin)):
                    intemp = datasetins[i + (look_back - look_backin[j]):i + (look_back), j]
                    for n in range(len(intemp)):
                        dataX.append(intemp[n])
                    dataX.append(datasetins[i + (look_back), j])
                DataX[i, :] = (dataX)
                DataY.append(datasetout[i + (look_back)])
            return DataX, np.array(DataY)

        trainX, trainY = create_TimeSeries(trainin,trainout[:,outPara], look_backin,look_backout)
        testX, testY = create_TimeSeries(testin,testinout[:,outPara], look_backin,look_backout)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        LSTMpointnum=80                     #80
        # create and fit the LSTM network
        model = Sequential()

        model.add(LSTM(LSTMpointnum, input_dim=feature))

        model.add(Dense(D1[test1]))                #40
        model.add(Activation('relu'))
        model.add(Dense(D2[test1]))                #20
        model.add(Activation('relu'))
        model.add(Dense(D3[test1]))                #10
        model.add(Activation('relu'))
        model.add(Dense(D4[test1]))                 #5
        model.add(Activation('relu'))
        model.add(Dense(1))                 #1
        model.add(Activation('sigmoid'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd)
        hist=model.fit(trainX, trainY, nb_epoch=1000, batch_size=1000, verbose=2, validation_split=0.3)
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
        # # plot baseline and predictions
        # loss=hist.history.get('loss')
        # plt.figure("loss")
        # plt.plot(loss)
        # plt.show()
        #
        # plt.figure()
        # plt.plot(range(look_back,len(realtrainPredict)+look_back,1),realtrainPredict,'r')
        #
        # plt.plot(range(len(realtrainPredict)+(look_back*2),sizeOutput[0],1),realtestPredict,'g')
        #
        # plt.plot(range(0,len(DataOutput[:,outPara])),DataOutput[:,outPara],'b')
        # plt.show()
        index1.append(test1)
        index2.append(test2)
        TRS.append(trainScore)
        TES.append(testScore)
np.save('LSTM_HPH/index1.npy',index1)
np.save('LSTM_HPH/index2.npy',index2)
np.save('LSTM_HPH/TRS.npy',TRS)
np.save('LSTM_HPH/TES.npy',TES)
sendEmail('LSTM Done')