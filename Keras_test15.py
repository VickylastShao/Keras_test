# coding: utf-8
# Q-Learning 矩阵解法的简单算理

import numpy as np

# QLearning：
# 1. 给定参数γ和R矩阵
# 2. 初始化 Q
# 3. for each episode:
# 3.1随机选择一个出事状态s
# 3.2若未达到目标状态，则执行以下几步
# (1)在当前状态s的所有可能行为中选取一个行为a
# (2)利用选定的行为a,得到下一个状态 。
# (3)按照 Q(s,a)=R(s,a)+γmax{Q(s^,a^)}
# (4) s:=s^
# γ 为学习参数，
# R为奖励机制， 为在s状态下，执行Q所得到的值。
# 随机选择一个一个状态，即开始搜索的起点，在为100的点为终点。

GAMMA = 0.8
Q = np.zeros((6,6))

R=np.asarray([[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,0.2],
   [-1,-1,-1,0,-1,-1],
   [-1,0, 0, -1,0,-1],
   [0,-1,-1,0,-1,0.2],
   [-1,0,-1,-1,0,0.2]])

def getMaxQ(state):
  return max(Q[state, :])

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
      print('getMaxQ(curAction): %.2f' % (getMaxQ(curAction)))
      print 'Q['+str(state)+','+str(action)+']='+str(R[state][action])+'+'\
            +str(GAMMA)+'*getMaxQ('+str(Q[state, :])+')'
      Q[state,action]=R[state][action]+GAMMA * getMaxQ(curAction)
      print('Q['+str(state)+','+str(action)+']: %.5f' % (Q[state,action]))
count=0
while count<1000:
    for i in xrange(6):
        QLearning(i)
    count+=1
print Q