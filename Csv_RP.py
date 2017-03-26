# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np


# for i in range(5):
#     if i==0:
#         model="111112"
#     elif i==1:
#         model = "111122"
#     elif i==2:
#         model = "111222"
#     elif i==3:
#         model = "112222"
#     elif i==4:
#         model = "122222"
#
#
#     loss=np.load('Share_Weight/loss_'+str(model)+'.npy')
#     acc=np.load('Share_Weight/acc_'+str(model)+'.npy')
#     testoutputL= np.loadtxt('Share_Weight/testoutputL_'+str(model)+'.csv', dtype=np.float,delimiter=",")
#     testoutputR= np.loadtxt('Share_Weight/testoutputR_'+str(model)+'.csv', dtype=np.float,delimiter=",")
#     resultRealL= np.loadtxt('Share_Weight/resultRealL_'+str(model)+'.csv', dtype=np.float,delimiter=",")
#     resultRealR= np.loadtxt('Share_Weight/resultRealR_'+str(model)+'.csv', dtype=np.float,delimiter=",")
#     errAbL = np.loadtxt('Share_Weight/errAbL_' + str(model) + '.csv', dtype=np.float, delimiter=",")
#     errAbR = np.loadtxt('Share_Weight/errAbR_' + str(model) + '.csv', dtype=np.float, delimiter=",")
#     errReL = np.loadtxt('Share_Weight/errReL_' + str(model) + '.csv', dtype=np.float, delimiter=",")
#     errReR = np.loadtxt('Share_Weight/errReR_' + str(model) + '.csv', dtype=np.float, delimiter=",")
#
#     plt.figure('loss_'+str(model))
#     plt.plot(loss)
#     plt.title('loss_'+str(model))
#     plt.savefig('loss_'+str(model)+'.jpg')
#
#
#     plt.figure('acc_'+str(model))
#     plt.plot(acc)
#     plt.title('acc_'+str(model))
#     plt.savefig('acc_'+str(model)+'.jpg')
#
#     plt.figure('testoutputL_' + str(model))
#     plt.plot(testoutputL)
#     plt.title('testoutputL_' + str(model))
#     plt.savefig('testoutputL_' + str(model) + '.jpg')
#
#     plt.figure('testoutputR_' + str(model))
#     plt.plot(testoutputR)
#     plt.title('testoutputR_' + str(model))
#     plt.savefig('testoutputR_' + str(model) + '.jpg')
#
#     plt.figure('resultRealL_' + str(model))
#     plt.plot(resultRealL)
#     plt.title('resultRealL_' + str(model))
#     plt.savefig('resultRealL_' + str(model) + '.jpg')
#
#     plt.figure('resultRealR_' + str(model))
#     plt.plot(resultRealR)
#     plt.title('resultRealR_' + str(model))
#     plt.savefig('resultRealR_' + str(model) + '.jpg')
#
#     plt.figure('errAbL_' + str(model))
#     plt.plot(errAbL)
#     plt.title('errAbL_' + str(model))
#     plt.savefig('errAbL_' + str(model) + '.jpg')
#
#     plt.figure('errAbR_' + str(model))
#     plt.plot(errAbR)
#     plt.title('errAbR_' + str(model))
#     plt.savefig('errAbR_' + str(model) + '.jpg')
#
#     plt.figure('errReL_' + str(model))
#     plt.plot(errReL)
#     plt.title('errReL_' + str(model))
#     plt.savefig('errReL_' + str(model) + '.jpg')
#
#     plt.figure('errReR_' + str(model))
#     plt.plot(errReR)
#     plt.title('errReR_' + str(model))
#     plt.savefig('errReR_' + str(model) + '.jpg')
#     plt.close('all')


ERL0,ERL1,ERL2,ERR0,ERR1,ERR2=[],[],[],[],[],[]
for i in range(5):
    # i=0
    if i==0:
        model="111112"
    elif i==1:
        model = "111122"
    elif i==2:
        model = "111222"
    elif i==3:
        model = "112222"
    elif i==4:
        model = "122222"
    errReL = np.loadtxt('Share_Weight/errReL_' + str(model) + '.csv', dtype=np.float, delimiter=",")
    errReR = np.loadtxt('Share_Weight/errReR_' + str(model) + '.csv', dtype=np.float, delimiter=",")

    ERL0.append(np.mean(np.abs(errReL)[:,0]))
    ERL1.append(np.mean(np.abs(errReL)[:,1]))
    ERL2.append(np.mean(np.abs(errReL)[:,2]))
    ERR0.append(np.mean(np.abs(errReR)[:,0]))
    ERR1.append(np.mean(np.abs(errReR)[:,1]))
    ERR2.append(np.mean(np.abs(errReR)[:,2]))

x=range(5)

plt.figure('ERL0')
plt.plot(x,ERL0)
plt.title('ERL0')
ax=plt.gca()
ax.set_xticks(range(5))
ax.set_xticklabels( ('111112', '111122', '111222', '112222', '122222'))
plt.savefig('ERL0.jpg')

plt.figure('ERL1')
plt.plot(x,ERL1)
plt.title('ERL1')
ax=plt.gca()
ax.set_xticks(range(5))
ax.set_xticklabels( ('111112', '111122', '111222', '112222', '122222'))
plt.savefig('ERL1.jpg')

plt.figure('ERL2')
plt.plot(x,ERL2)
plt.title('ERL2')
ax=plt.gca()
ax.set_xticks(range(5))
ax.set_xticklabels( ('111112', '111122', '111222', '112222', '122222'))
plt.savefig('ERL2.jpg')


plt.figure('ERR0')
plt.plot(x,ERR0)
plt.title('ERR0')
ax=plt.gca()
ax.set_xticks(range(5))
ax.set_xticklabels( ('111112', '111122', '111222', '112222', '122222'))
plt.savefig('ERR0.jpg')

plt.figure('ERR1')
plt.plot(x,ERR1)
plt.title('ERR1')
ax=plt.gca()
ax.set_xticks(range(5))
ax.set_xticklabels( ('111112', '111122', '111222', '112222', '122222'))
plt.savefig('ERR1.jpg')

plt.figure('ERR2')
plt.plot(x,ERR2)
plt.title('ERR2')
ax=plt.gca()
ax.set_xticks(range(5))
ax.set_xticklabels( ('111112', '111122', '111222', '112222', '122222'))
plt.savefig('ERR2.jpg')



