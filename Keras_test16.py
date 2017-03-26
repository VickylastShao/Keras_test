# coding: utf-8
# Deep Q Network 简单的神经网络实现

from Keras_Defs import *

GAMMA = 0.8
Q = np.zeros((6,6))

R=np.asarray([[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,0.2],
   [-1,-1,-1,0,-1,-1],
   [-1,0, 0, -1,0,-1],
   [0,-1,-1,0,-1,0.2],
   [-1,0,-1,-1,0,0.2]])

model=Sequential()
model.add(Dense(32, init='uniform', input_dim=1))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)
train=np.zeros((6,1))
lable=np.zeros((6,6))

for i in range(6):
    train[i, 0] = float(i)/5
    for j in range(6):
        lable[i,j]=0
model.fit(train, lable, batch_size=6, nb_epoch=1000)
# Q模型1输入6输出，输入当前状态，输出6个动作的收益值
print model.predict(train)

def getMaxQ(state):
    pre = np.zeros((1,1))
    pre[0,0]=float(state)/5
    outQ=model.predict(pre)
    return np.max(outQ[0,:])

def QLearning(state):
    curAction = None
    for action in xrange(6):
        if(R[state][action] == -1):
            Q[state, action]=0
        else:
            curAction = action
            print '-----------------count'+str(count)+'-------------------'
            print('state: %.0f' % (state))
            print('action: %.0f' % (action))
            print('R['+str(state)+','+str(action)+']: %.0f' % (R[state][action]))
            print('GAMMA: %.1f' % (GAMMA))
            # print('getMaxQ(curAction): %.2f' % (getMaxQ(curAction)))
            print 'Q['+str(state)+','+str(action)+']='+str(R[state][action])+'+'\
                    +str(GAMMA)+'*getMaxQ('+str(Q[state, :])+')'
            Q[state,action]=R[state][action]+GAMMA * getMaxQ(curAction)
            print('Q['+str(state)+','+str(action)+']: %.5f' % (Q[state,action]))
    return Q[state,:]


count=0
# i=0 #初始化一个状态
while count<100:
    #计算该状态下的所有action对应的收益
    Qstate = np.zeros((6, 6))#用来存储一批训练样本
    for i in range(6):
        Qstate[i,:]=QLearning(i)#在状态i生成1组训练样本，输入i，输出6个动作的收益值
    Traindata = np.zeros((6, 1))
    Labledata = np.zeros((6, 6))
    for i in range(6):
        Traindata[i, 0] = float(i)/5
        for j in range(6):
            Labledata[i, j] = Qstate[i, j]
    hist =model.fit(Traindata, Labledata, batch_size=6, nb_epoch=500)
    # loss = hist.history.get('loss')
    # plt.figure("loss")
    # plt.plot(loss)
    # plt.show()
    count+=1
print Q


