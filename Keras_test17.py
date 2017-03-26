# coding: utf-8
# 连续问题的 Deep Q Network 状态连续 动作离散
# 状态连续0~1,动作为-(actionKind-1)/2到(actionKind-1)/2共actionKind个动作
# 分析:
# 对于R矩阵,原始的i行状态,j列动作,连续问题时改成某一分布函数
# 输入[状态，动作],输出即时奖励r
# 对于Q矩阵,原始的输入一个状态值，输出所有actionKind个动作的Q值

from Keras_Defs import *

GAMMA = 0.4
actionKind=101#动作的种类数量，奇数比较好(包含动作0)，默认在0附近均匀分布


# 读取R矩阵,获取即时奖励
def getR(state,action):
    # 状态的即时奖励函数:为一正弦曲线,state为0.5时奖励为1,state为0和1时奖励为0
    R1=math.sin(state*(math.pi))
    # 动作的即时奖励
    # 该动作若使状态靠近0.5,则奖励为8*(靠近量^3)
    # (靠近量范围为0~0.5,折合到奖励为0~1)
    # 若靠近量为负,奖励为0
    R2=0
    if abs(state-0.5)-abs((state+action)-0.5)>0:
        R2=8*(abs(state-0.5)-abs((state+action)-0.5))**3
    #两个即时奖励加权
    R=0.1*R1+0.9*R2
    return R

# 初始化全零Q模型
model=Sequential()
model.add(Dense(64, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(actionKind))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)
train=np.zeros((21,1))
lable=np.zeros((21,actionKind))
for i in range(21):
    train[i, 0] = float(i)/20 #当前状态0~1连续遍历
    for j in range(actionKind):
        lable[i,j]=0#所有5个动作的收益都初始化为0
model.fit(train, lable, batch_size=21, nb_epoch=1000)
# Q模型1输入5输出，输入当前状态，输出actionKind个动作的收益值


# 获取当前状态state下执行actionKind个动作后最高的[收益值,索引]
def getMaxQ(state,model):
    statein = np.zeros((1, 1))
    statein[0,0]=state
    outQ=model.predict(statein)
    outQlist=outQ[0,:].tolist()
    maxQ=max(outQlist)#求所有动作收益的最大值
    maxindex=outQlist.index(max(outQlist))#查找最大值索引
    out=[maxQ,maxindex]
    return out

# 最小动作步长
stepLength=0.0002

# 获取当前状态state执行actionKind个动作之后的收益
def QLearning(state,model):
    Q = np.zeros((actionKind))
    for i in range(actionKind):
        # actionKind个动作遍历
        action=(float(i)-(actionKind-1)/2) * stepLength
        # 获得即时奖励
        R = getR(state,action)
        # 当前状态state下执行当前动作action,到达新状态NextState
        NextState = state+action
        # 新状态NextState超限判断
        if NextState>1:
            NextState=1
        elif NextState<0:
            NextState=0
        # 获取新状态下actionKind个动作的最大收益
        nextQ=getMaxQ(NextState,model)[0]
        # Q学习迭代公式
        Q[i]=(1-GAMMA)*R+GAMMA * nextQ
    return Q

count=0
update=50#Q模型更新次数
while count<update:
    #计算该状态下的所有action对应的收益
    samples=50#每次模型更新的样本数量
    Qstate = np.zeros((samples, 1))  # 存储一批训练样本的状态
    Qvalue = np.zeros((samples, actionKind))  # 存储一批训练样本actionKind个动作的收益值
    for i in range(samples):
        curstate=random.uniform(0, 1)
        Qstate[i,0]=curstate # 生成随机状态i
        Qvalue[i,:]=QLearning(float(curstate),model)#在状态i生成1组训练样本state->[value1,value2...]
    hist=model.fit(Qstate, Qvalue, batch_size=50, nb_epoch=1000,validation_split=0.1)
    # loss = hist.history.get('loss')
    # plt.figure("loss")
    # plt.plot(loss)
    # plt.show()
    count+=1


# 开始运行
path=[]#最优路线
startnum=11#0~1起点数量
actionnum=200#动作次数
for ss in range(startnum):
    StartState=float(ss)/(startnum-1)#起点为0,0.1,...0.9,1
    states=[]
    states.append(StartState)
    for i in range(actionnum):
        # 根据模型,求得最优动作的索引
        maxActionIndex=getMaxQ(states[-1],model)[1]
        # 求动作
        action = (float(maxActionIndex) - (actionKind-1)/2) * stepLength
        # 新状态
        NextState = states[-1] + action
        # 新状态NextState超限判断
        if NextState > 1:
            NextState = 1
        elif NextState < 0:
            NextState = 0
        states.append(NextState)
    path.append(states)

# 绘图
plt.figure()
for i in range(len(path[:])):
    plt.plot(path[i][:])
plt.show()
print path









