# coding: utf-8
# GANs自编
# 模型输入为2个随机变量

from Keras_Defs import *

def getRandom(n,m):
    Random=np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Random[i,j] = random.uniform(0, 1)
    return Random

def getRandomFakedata(n):
    RandomFakedata = np.zeros((n, 2))
    for i in range(n):
        RandomFakedata[i, 0] = random.uniform(0, 1)
        RandomFakedata[i, 1] = random.uniform(0, 1)
    return RandomFakedata

def getRealdata(n):
    Realdata = np.zeros((n, 2))
    for i in range(n):
        R = random.uniform(0, 1)
        Realdata[i, 0] = R
        # Realdata[i, 1] = math.sqrt(1 - R ** 2)+random.uniform(0, 1)*0.05
        Realdata[i, 1] = 1-R + (random.uniform(0, 1)-0.5) * 0.1
    return Realdata

def getFakedata(n,GeneModel):
    Rs = getRandom(n,2)
    Fakedata=GeneModel.predict(Rs)
    return Fakedata

## Plotting
def WebPlot(num_points,GeneModel_M):
    fake_data_batch=getFakedata(num_points,GeneModel_M)
    data_batch=getRealdata(num_points)

    # Plot distributions
    trace_fake = go.Scatter(
        x = fake_data_batch[:,0],
        y = fake_data_batch[:,1],
        mode = 'markers',
        name='Generated Data'
    )

    trace_real = go.Scatter(
        x = data_batch[:,0],
        y = data_batch[:,1],
        mode = 'markers',
        name = 'Real Data'
    )

    data = [trace_fake, trace_real]
    fig = go.Figure(data=data)
    py.plot(fig)


# step1:
# 初始化GANs模型、生成模型、判别模型

# 初始化GANs整体模型
# GANs生成模型层 输入2 输出2
GANs_Input=Input(shape=(2,),name='GANs_Input')
GANs=Dense(16,activation='relu',name='GANs_G1')(GANs_Input)
GANs=Dense(16,activation='relu',name='GANs_G2')(GANs)
GANs=Dense(2,activation='sigmoid',name='GANs_G3')(GANs)
# GANs判别模型层 输入2 输出1
GANs=Dense(32,activation='relu',name='GANs_D1')(GANs)
GANs=Dense(32,activation='relu',name='GANs_D2')(GANs)
GANs=Dense(32,activation='relu',name='GANs_D3')(GANs)
GANs=Dense(32,activation='relu',name='GANs_D4')(GANs)
GANs_Output=Dense(1,activation='sigmoid',name='GANs_D5')(GANs)

GANs_M = Model(input=[GANs_Input], output=[GANs_Output])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
GANs_M.compile(loss='binary_crossentropy', optimizer=sgd)


# 初始化生成模型
GeneModel_Input=Input(shape=(2,),name='GeneModel_Input')
GeneModel=Dense(16,activation='relu',name='G1')(GeneModel_Input)
GeneModel=Dense(16,activation='relu',name='G2')(GeneModel)
GeneModel_Output=Dense(2,activation='sigmoid',name='G3')(GeneModel)
GeneModel_M=Model(input=[GeneModel_Input], output=[GeneModel_Output])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
GeneModel_M.compile(loss='binary_crossentropy', optimizer=sgd)

# 初始化判别模型
DiscModel_Input=Input(shape=(2,),name='DiscModel_Input')
DiscModel=Dense(32,activation='relu',name='D1')(DiscModel_Input)
DiscModel=Dense(32,activation='relu',name='D2')(DiscModel)
DiscModel=Dense(32,activation='relu',name='D3')(DiscModel)
DiscModel=Dense(32,activation='relu',name='D4')(DiscModel)
DiscModel_Output=Dense(1,activation='sigmoid',name='D5')(DiscModel)
DiscModel_M=Model(input=[DiscModel_Input], output=[DiscModel_Output])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
DiscModel_M.compile(loss='binary_crossentropy', optimizer=sgd)

avgGout=[]
testGout = GeneModel_M.predict(getRandom(10,2))
avgGout.append(sum(testGout[:,0] + testGout[:,1] )/10.0)

for i in range(2000):
    # step2:训练判别模型
    # 训练样本1:真实数据,lable=1;训练样本2:生成模型输出数据,lable=0
    rn=50 # 真训练样本个数
    fn=50 # 假训练样本个数
    if i==0:
        Realdata = getRealdata(rn)
        Fakedata = getRandomFakedata(fn)
    else:
        Realdata = getRealdata(rn)
        Fakedata = getFakedata(fn,GeneModel_M)
    Reallable=np.ones((rn,1))
    Fakelable=np.zeros((fn,1))
    Disctrain=np.concatenate((Realdata,Fakedata))
    Disclable=np.concatenate((Reallable,Fakelable))
    # 模型训练
    hist = DiscModel_M.fit(Disctrain, Disclable, batch_size=200, nb_epoch=10, validation_split=0.1)
    # loss = hist.history.get('loss')
    # plt.figure("loss")
    # plt.plot(loss)
    # plt.show()
    # 将判别模型权重赋给GANs模型
    GANs_M.get_layer(name='GANs_D1').set_weights(DiscModel_M.get_layer(name='D1').get_weights())
    GANs_M.get_layer(name='GANs_D2').set_weights(DiscModel_M.get_layer(name='D2').get_weights())
    GANs_M.get_layer(name='GANs_D3').set_weights(DiscModel_M.get_layer(name='D3').get_weights())
    GANs_M.get_layer(name='GANs_D4').set_weights(DiscModel_M.get_layer(name='D4').get_weights())
    GANs_M.get_layer(name='GANs_D5').set_weights(DiscModel_M.get_layer(name='D5').get_weights())
    # 将生成模型权重赋给GANs模型
    GANs_M.get_layer(name='GANs_G1').set_weights(GeneModel_M.get_layer(name='G1').get_weights())
    GANs_M.get_layer(name='GANs_G2').set_weights(GeneModel_M.get_layer(name='G2').get_weights())
    GANs_M.get_layer(name='GANs_G3').set_weights(GeneModel_M.get_layer(name='G3').get_weights())

    # step3:训练生成模型
    # 冻结GANs判别层权重
    GANs_M.get_layer(name='GANs_D1').trainable = False
    GANs_M.get_layer(name='GANs_D2').trainable = False
    GANs_M.get_layer(name='GANs_D3').trainable = False
    GANs_M.get_layer(name='GANs_D4').trainable = False
    GANs_M.get_layer(name='GANs_D5').trainable = False
    # 训练样本:随机数,lable=1
    Gn=20 # 训练样本个数
    GANstrain=getRandom(Gn,2)
    GANslable=np.ones((Gn,1))
    # 模型训练
    # if i==0 or i%1==0:
    hist = GANs_M.fit(GANstrain, GANslable, batch_size=200, nb_epoch=3, validation_split=0.1)
    # loss = hist.history.get('loss')
    # plt.figure("loss")
    # plt.plot(loss)
    # plt.show()
    # 将GANs模型生成层权重赋给生成模型
    GeneModel_M.get_layer(name='G1').set_weights(GANs_M.get_layer(name='GANs_G1').get_weights())
    GeneModel_M.get_layer(name='G2').set_weights(GANs_M.get_layer(name='GANs_G2').get_weights())
    GeneModel_M.get_layer(name='G3').set_weights(GANs_M.get_layer(name='GANs_G3').get_weights())

    # 绘图
    testGout = getFakedata(10,GeneModel_M)
    avgGout.append(sum(testGout[:, 0] + testGout[:, 1] ) / 10.0)
    if i%100==0:
        WebPlot(500, GeneModel_M)


plt.figure('avgGout')
plt.plot(avgGout)
plt.show()
fakedata= getFakedata(10,GeneModel_M)
print fakedata
print DiscModel_M.predict(fakedata)




