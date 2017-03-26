# coding: utf-8
# 连续问题的 Deep Q Network 状态连续 动作离散
# 仿真汽包水位的控制逻辑,并用DQN来实现
# 建模部分


from Keras_Defs import *

GAMMA=0.96

actionKind=101 #动作的种类数量，奇数比较好(包含动作0)，默认在0附近均匀分布

# 初始化全零Q模型
model=Sequential()
model.add(Dense(64, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(actionKind))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)
train=np.zeros((21*21,2))
lable=np.zeros((21*21,actionKind))
for i1 in range(21):
    for i2 in range(21):
        train[i1*21+i2, 0] = float(i1) / 20  # 当前状态Le0~1连续遍历
        train[i1*21+i2, 1] = float(i2) / 20  # 当前状态Po0~1连续遍历
        for j in range(actionKind):
            lable[i1*21+i2,j]=0#所有actionKind个动作的收益都初始化为0
model.fit(train, lable, batch_size=21*21, nb_epoch=1000)
# Q模型1输入5输出，输入当前状态，输出actionKind个动作的收益值

def getNextState(state,action):
    curLe=state[0]
    curPo=state[1]
    nextPo=curPo+action
    if nextPo>1:
        nextPo=1.0
    elif nextPo<0:
        nextPo=0.0
    inputMass=10.0
    # 密度*截面积
    Area=1000.0
    # 阀门阻力特性
    outputMass=nextPo*20.0
    deltaMass=inputMass-outputMass
    deltaLe=deltaMass/Area
    nextLe=curLe+deltaLe
    if nextLe>1:
        nextLe=1.0
    elif nextLe<0:
        nextLe=0.0
    return [nextLe,nextPo]


# 读取R矩阵,获取即时奖励
def getR(state,action):
    # 状态1的即时奖励函数:水位越接近0.5越好 0~1
    # R11 = math.sin(state[0]*(math.pi))
    R11 = 25*(min((state[0])**5,(1-state[0])**5))

    # 状态2的即时奖励函数:开度与即时奖励无关
    # R12 = 0.0

    # # 动作的即时奖励:
    # if (state[0]>0.5 and action>0) or (state[0]<0.5 and action<0):
    #     R2 = 0.0
    # else:
    #     R2 = 1.0

    #两个即时奖励加权
    # R=0.8*(R11+R12)+0.2*R2
    R=R11
    return R


# 获取当前状态state下执行5个动作后最高的[收益值,索引]
def getMaxQ(state,model):
    outQ=model.predict(state)
    maxQ=np.zeros((len(outQ)))
    for i in range(len(outQ)):
        maxQ[i]=max(outQ[i,:].tolist())#求所有动作收益的最大值
    return maxQ

def getMaxQIndex(state,model):
    outQ=model.predict(state)
    maxindex=np.zeros((len(outQ)))


    for i in range(len(outQ)):
        outQlist = outQ[i, :].tolist()
        maxindex[i]=outQlist.index(max(outQlist))#查找最大值索引
    return maxindex

# 最小动作步长
stepLength=0.0002

# 获取当前状态state执行actionKind个动作之后的收益
def QLearning(Qstate,model):
    Qvalue = np.zeros((len(Qstate),actionKind))
    R = np.zeros((len(Qstate),actionKind))
    action= np.zeros((len(Qstate),actionKind))
    NextState=np.zeros((len(Qstate)*actionKind,2))
    for n in range(len(Qstate)):
        for i in range(actionKind):
            # actionKind个动作遍历
            action[n,i]=(float(i)-(actionKind-1)/2) * stepLength
            # 获得即时奖励
            R[n,i] = getR(Qstate[n,:],action[n,i])
            # 当前状态state下执行当前动作action,到达新状态NextState
            NextState[n*actionKind+i,:]=getNextState(Qstate[n,:],action[n,i])
            # 获取新状态下actionKind个动作的最大收益
    nextQ=getMaxQ(NextState,model)
    # Q学习迭代公式
    for n in range(len(Qstate)):
        for i in range(actionKind):
            Qvalue[n,i]=(1-GAMMA)*R[n,i]+GAMMA * nextQ[n*actionKind+i]
    return Qvalue

count=0
update=300#Q模型更新次数
while count<update:
    #计算该状态下的所有action对应的收益
    samples=400#每次模型更新的样本数量

    Qstate = np.zeros((samples, 2))  # 存储一批训练样本的状态
    Qvalue = np.zeros((samples, actionKind))  # 存储一批训练样本actionKind个动作的收益值
    for i in range(samples):
        curLe = random.uniform(0, 1)
        curPo = random.uniform(0, 1)
        Qstate[i, 0] = curLe # 生成随机水位状态Le
        Qstate[i, 1] = curPo # 生成随机开度状态Po
    # starttime = datetime.datetime.now()23
    Qvalue = QLearning(Qstate,model)#在状态i生成1组训练样本state->[value1,value2...]
    # hist=model.fit(Qstate, Qvalue, batch_size=400, nb_epoch=2000,validation_split=0.1)
    hist = model.fit(Qstate, Qvalue, batch_size=samples, nb_epoch=2000, validation_split=0.1)

    count+=1



# 开始运行
path = []  # 最优路线
startnum = 51  # 0~1起点数量
actionnum = 1000  # 动作次数
for ss in range(startnum):
    StartLe = float(ss) / (startnum - 1)  # 起点水位为0,0.1,...0.9,1
    StartPo = 0.5
    states = []
    states.append([StartLe, StartPo])
    for i in range(actionnum):
        # 根据模型,求得最优动作的索引
        statenow = np.zeros((1, 2))
        statenow[0, :] = states[-1]
        maxActionIndex = getMaxQIndex(statenow, model)
        # 求动作
        action = (float(maxActionIndex) - (actionKind - 1) / 2) * stepLength
        # 新状态
        NextState = getNextState(states[-1], action)
        states.append(NextState)
    path.append(states)

# 绘图
result1 = np.zeros((len(path[:]), len(path[0][:])))
for i in range(len(path[:])):
    for j in range(len(path[0][:])):
        result1[i, j] = path[i][j][0]
plt.figure()
for i in range(len(result1[:, 0])):
    plt.plot(result1[i, :])
plt.savefig('DQN/model_updatecount_' + str(count) + '-1.png')

result2 = np.zeros((len(path[:]), len(path[0][:])))
for i in range(len(path[:])):
    for j in range(len(path[0][:])):
        result2[i, j] = path[i][j][1]
plt.figure()
for i in range(len(result1[:, 0])):
    plt.plot(result2[i, :])
plt.savefig('DQN/model_updatecount_' + str(count) + '-2.png')
model.save('DQN/model_updatecount_' + str(count) + '.h5')

sendEmail('DQN Done')







