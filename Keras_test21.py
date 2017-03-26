# coding: utf-8
# CGANs自编
# 模型输入为2个随机变量 外加1个条件变量

from Keras_Defs import *
from keras.utils.visualize_util import plot


def getRealz(n):
    Random = np.zeros((n,1))
    for i in range(n):
        Random[i] = random.uniform(0, 1)
    return Random

def getFakez(n):
    return getRealz(n)

def getRandom(n,m):
    Random=np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Random[i,j] = random.uniform(0, 1)
    return Random

def getRandomFakedata(n,Fakez):
    RandomFakedata = np.zeros((n, 2))
    for i in range(n):
        RandomFakedata[i, 0] = random.uniform(0, 1)
        RandomFakedata[i, 1] = random.uniform(0, 1)
    return RandomFakedata

def getRealdata(n,Realz):
    Realdata = np.zeros((n, 2))
    for i in range(n):
        # R = random.uniform(0, 1)-0.5
        Realdata[i, 0] = Realz[i] + (random.uniform(0, 1)-0.5) * 0.1
        # Realdata[i, 1] = math.sqrt(1 - R ** 2)+random.uniform(0, 1)*0.05
        Realdata[i, 1] = Realz[i] + (random.uniform(0, 1)-0.5) * 0.1
    return Realdata

def getFakedata(n,Fakez,GeneModel):
    Rs = getRandom(n,2)
    Fakedata=GeneModel.predict([Rs,Fakez])
    return Fakedata

## Plotting
def WebPlot(num_points,GeneModel_M):
    Fakez = getFakez(num_points)
    fake_data_batch=getFakedata(num_points,Fakez,GeneModel_M)
    Realz = getRealz(num_points)
    data_batch=getRealdata(num_points,Realz)

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
# 初始化CGANs模型、生成模型、判别模型

# 初始化CGANs整体模型
# CGANs生成模型层 输入2+1 输出2
GANs_Input_z=Input(shape=(1,),name='GANs_Input_z')
GANs_Input_r=Input(shape=(2,),name='GANs_Input_r')
GANs_Merged1=merge([GANs_Input_r,GANs_Input_z],mode='concat')
GANs=Dense(16,activation='relu',name='GANs_G1')(GANs_Merged1)
# GANs=Dense(16,activation='relu',name='GANs_G2')(GANs)
GANs=Dense(2,activation='sigmoid',name='GANs_G2')(GANs)
# CGANs判别模型层 输入2 输出1
GANs_Merged2=merge([GANs,GANs_Input_z],mode='concat',concat_axis=1)
GANs=Dense(32,activation='relu',name='GANs_D1')(GANs_Merged2)
GANs=Dense(32,activation='relu',name='GANs_D2')(GANs)
GANs=Dense(32,activation='relu',name='GANs_D3')(GANs)
GANs=Dense(32,activation='relu',name='GANs_D4')(GANs)
GANs_Output=Dense(1,activation='sigmoid',name='GANs_D5')(GANs)

GANs_M = Model(input=[GANs_Input_r,GANs_Input_z], output=[GANs_Output])
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
GANs_M.compile(loss='binary_crossentropy', optimizer=sgd)

# plot(GANs_M, to_file='GANs_M.png')

# 初始化生成模型
GeneModel_Input_z=Input(shape=(1,),name='GeneModel_Input_z')
GeneModel_Input_r=Input(shape=(2,),name='GeneModel_Input_r')
GeneModel_Merged=merge([GeneModel_Input_r,GeneModel_Input_z],mode='concat')
GeneModel=Dense(16,activation='relu',name='G1')(GeneModel_Merged)
# GeneModel=Dense(16,activation='relu',name='G2')(GeneModel)
GeneModel_Output=Dense(2,activation='sigmoid',name='G2')(GeneModel)
GeneModel_M=Model(input=[GeneModel_Input_r,GeneModel_Input_z], output=[GeneModel_Output])
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
GeneModel_M.compile(loss='binary_crossentropy', optimizer=sgd)

# 初始化判别模型
DiscModel_Input_z=Input(shape=(1,),name='DiscModel_Input_z')
DiscModel_Input_r=Input(shape=(2,),name='DiscModel_Input_r')
DiscModel_Merged=merge([DiscModel_Input_r,DiscModel_Input_z],mode='concat')
DiscModel=Dense(32,activation='relu',name='D1')(DiscModel_Merged)
DiscModel=Dense(32,activation='relu',name='D2')(DiscModel)
DiscModel=Dense(32,activation='relu',name='D3')(DiscModel)
DiscModel=Dense(32,activation='relu',name='D4')(DiscModel)
DiscModel_Output=Dense(1,activation='sigmoid',name='D5')(DiscModel)
DiscModel_M=Model(input=[DiscModel_Input_r,DiscModel_Input_z], output=[DiscModel_Output])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
DiscModel_M.compile(loss='binary_crossentropy', optimizer=sgd)


avgGout=[]
Fakez = getFakez(10)
testGout = getFakedata(10,Fakez,GeneModel_M)
avgGout.append(sum(testGout[:,0] - testGout[:,1] )/10.0)

for i in range(10000):
    # step2:训练判别模型
    # 训练样本1:真实数据,lable=1;训练样本2:生成模型输出数据,lable=0
    rn=500 # 真训练样本个数
    fn=500 # 假训练样本个数
    if i==0:
        Realz = getRealz(rn)
        Realdata = getRealdata(rn,Realz)
        Fakez = getFakez(fn)
        Fakedata = getRandomFakedata(fn,Fakez)
    else:
        Realz = getRealz(rn)
        Realdata = getRealdata(rn,Realz)
        Fakez = getFakez(fn)
        Fakedata = getFakedata(fn,Fakez,GeneModel_M)
    Reallable=np.ones((rn,1))
    Fakelable=np.zeros((fn,1))


    Disctrain=[np.concatenate((Realdata,Fakedata)),np.concatenate((Realz, Fakez))]
    Disclable=np.concatenate((Reallable,Fakelable))
    # 模型训练
    hist = DiscModel_M.fit(Disctrain, Disclable, batch_size=rn+fn, nb_epoch=7, validation_split=0.1)
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
    # GANs_M.get_layer(name='GANs_G2').set_weights(GeneModel_M.get_layer(name='G2').get_weights())
    GANs_M.get_layer(name='GANs_G2').set_weights(GeneModel_M.get_layer(name='G2').get_weights())

    # step3:训练生成模型
    # 冻结GANs判别层权重
    GANs_M.get_layer(name='GANs_D1').trainable = False
    GANs_M.get_layer(name='GANs_D2').trainable = False
    GANs_M.get_layer(name='GANs_D3').trainable = False
    GANs_M.get_layer(name='GANs_D4').trainable = False
    GANs_M.get_layer(name='GANs_D5').trainable = False
    # 训练样本:随机数,lable=1
    Gn=200 # 训练样本个数
    GANstrain=getRandom(Gn,2)
    GANstrainz=getRandom(Gn,1)
    GANslable=np.ones((Gn,1))
    # 模型训练
    # if i==0 or i%1==0:
    hist = GANs_M.fit([GANstrain,GANstrainz], GANslable, batch_size=Gn, nb_epoch=3, validation_split=0.1)
    # loss = hist.history.get('loss')
    # plt.figure("loss")
    # plt.plot(loss)
    # plt.show()
    # 将CGANs模型生成层权重赋给生成模型
    GeneModel_M.get_layer(name='G1').set_weights(GANs_M.get_layer(name='GANs_G1').get_weights())
    # GeneModel_M.get_layer(name='G2').set_weights(GANs_M.get_layer(name='GANs_G2').get_weights())
    GeneModel_M.get_layer(name='G2').set_weights(GANs_M.get_layer(name='GANs_G2').get_weights())

    # 绘图
    Fakez = getFakez(10)
    testGout = getFakedata(10,Fakez,GeneModel_M)
    avgGout.append(sum(testGout[:, 0] - testGout[:, 1] ) / 10.0)
    if i%100==0:
        WebPlot(500, GeneModel_M)


plt.figure('avgGout')
plt.plot(avgGout)
plt.show()
# Fakez = getFakez(10)
# fakedata= getFakedata(10,Fakez,GeneModel_M)
# print fakedata
# print DiscModel_M.predict([fakedata,Fakez])




