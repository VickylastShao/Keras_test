# coding: utf-8
# 岱海引风机数据

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model

model=load_model('DH_fan_model.h5')
data=np.load('data.npy')
input=np.load('input.npy')
output=np.load('output.npy')
train=np.load('train.npy')
lable=np.load('lable.npy')
trainsamplenum=np.load('trainsamplenum.npy')
inputnum=np.load('inputnum.npy')
outputnum=np.load('outputnum.npy')



print "3D fig"
# predict with normalized data
X=np.arange(0,1,0.05)
Y=np.arange(0,1,0.05)
X,Y=np.meshgrid(X,Y)
arr=np.zeros((len(X)*len(Y),2))
m=0
halftemparr=0.5*np.ones((len(X)*len(Y),inputnum-2))
for i in range (len(X)):
    for j in range(len(Y)):
        arr[m] = np.hstack((X[i][j], Y[i][j]))
        m+=1
arr = np.hstack((arr, halftemparr))
result = model.predict(arr, batch_size=1)
Z=np.zeros((len(X),len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i][j] = result[i*len(Y)+j,0]

# Anti normalization
x=np.zeros((len(X),len(Y)))
y=np.zeros((len(X),len(Y)))
z=np.zeros((len(X),len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        x[i,j] = X[i, j] * (np.max(input[:, 0]) - np.min(input[:, 0])) + np.min(input[:, 0])

        y[i,j] = Y[i, j] * (np.max(input[:, 1]) - np.min(input[:, 1])) + np.min(input[:, 1])

        z[i,j]=Z[i,j]*(np.max(output[:,0])-np.min(output[:,0]))+np.min(output[:,0])

fig=plt.figure()
ax=Axes3D(fig)
ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='rainbow')
plt.show()
# np.savetxt('resultx.csv', x, delimiter = ',')
# np.savetxt('resulty.csv', y, delimiter = ',')
# np.savetxt('resultz.csv', z, delimiter = ',')

print("predict")
predictnum=300
#真实数据作为测试输入
testinput=data[trainsamplenum+1:trainsamplenum+1+predictnum,:inputnum]# 0~7输入 8~18输出
#真实数据作为输出，用来对比效果
testoutput=data[trainsamplenum+1:trainsamplenum+1+predictnum,inputnum:]
#归一化
test=np.zeros((predictnum,inputnum))
for i in range(predictnum):
    for j in range(inputnum):
        test[i,j]=(testinput[i,j]-np.min(input[:,j]))/(np.max(input[:,j])-np.min(input[:,j]))

#得到模型输出(归一化的)
testresult = model.predict(test, batch_size=1)

#将模型输出进行反归一化
testresultReal=np.zeros((predictnum,outputnum))
for i in range(predictnum):
    for j in range(outputnum):
        testresultReal[i,j]=testresult[i,j]*(np.max(output[:,j])-np.min(output[:,j]))+np.min(output[:,j])

for i in range(outputnum):
    err1=testresultReal[:,i]-testoutput[:,i]#模型输出-实际输出
    err2 = err1  / (np.max(output[:, i]) - np.min(output[:, i])) # 相对预测误差
    xaxis=range(predictnum)
    plt.figure(i) # 创建图表
    ax1 = plt.subplot(221)  # 在图表中创建子图1
    ax2 = plt.subplot(222)  # 在图表中创建子图2
    ax3 = plt.subplot(223)  # 在图表中创建子图3
    ax4 = plt.subplot(224)  # 在图表中创建子图4
    plt.sca(ax1)   # 选择图表的子图1
    plt.plot(xaxis, err1)
    plt.title("absolute error " + str(i))
    plt.sca(ax2)  # 选择图表的子图2
    plt.plot(xaxis, err2)
    plt.title("related error "+str(i))
    plt.sca(ax3)  # 选择图表的子图3
    plt.plot(xaxis, testresultReal[:,i])
    plt.title("model output " + str(i))
    plt.sca(ax4)  # 选择图表的子图4
    plt.plot(xaxis, testoutput[:,i])
    plt.title("real output " + str(i))
    plt.show()