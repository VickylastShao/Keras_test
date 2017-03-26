# coding: utf-8
# 读取几个CSV表的数据并整合到一个新的CSV表格中


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import smtplib

# for m in range(3):
#     if m == 0:
#         datanum= 200000
#     elif m == 1:
#         datanum= 250000
#     elif m == 2:
#         datanum = 300000
#     for n in range(6):
#         if n == 0:
#             opnum = 3
#         elif n == 1:
#             opnum = 10
#         elif n == 2:
#             opnum = 11
#         elif n == 3:
#             opnum = 13
#         elif n == 4:
#             opnum = 15
#         elif n == 5:
#             opnum = 19
#
#         Data=np.arange(30001*9/1.0).reshape(30001,9)
#         for i in range(9):
#             dropout=(i+1)/10.0
#             OriData = np.loadtxt('Simple_models/avgerr_model_0_'+str(datanum)+'_pre_start_end_5L_output['+str(opnum)+']_Dropout'+str(dropout)+'.csv', dtype=np.float, delimiter=",")
#             Data[:,i]=np.concatenate(([dropout],OriData))
#
#         np.savetxt('Share_Weight/Share_Weight_'+str(opnum)+'_'+str(datanum)+'.csv', Data, delimiter = ',')
#         print m,'-',n


# for m in range(2):
#     if m == 0:
#         datanum= 200000
#     elif m == 1:
#         datanum= 250000
#     # elif m == 2:
#     #     datanum = 300000
#     for n in range(6):
#         if n == 0:
#             opnum = 3
#         elif n == 1:
#             opnum = 10
#         elif n == 2:
#             opnum = 11
#         elif n == 3:
#             opnum = 13
#         elif n == 4:
#             opnum = 15
#         elif n == 5:
#             opnum = 19
#
#         avgerrlist=np.loadtxt('Share_Weight/Share_Weight_'+str(opnum)+'_'+str(datanum)+'.csv', dtype=np.float, delimiter=",")
#         avgerr=avgerrlist[1:,:]
#         a=[]
#         for i in range(9):
#             a.append(np.mean(np.abs(avgerr[datanum/10:29999,i])))
#         plt.figure()
#         plt.plot(a)
#         plt.title('RE_' + str(opnum) + '_' + str(datanum))
#         plt.savefig('Share_Weight/RE_' + str(opnum) + '_' + str(datanum) +'.jpg')
#         print m,'-',n



# xaxis=range(len(avgerr))
# plt.figure("errRe")
# plt.plot(xaxis[0:9999], avgerr[20000:29999,8])
# plt.title("related error")
# plt.show()


# Data=np.zeros((12+2,55))
# i=0
# for mn in range(11-1):
#     midnum = 11-mn
#     for cn in range(midnum-1):
#         codenum=midnum-cn-1
#
#         OriData = np.loadtxt('AutoEncoder/AutoEncoder_mid'+str(midnum)+'_code'+str(codenum)+'.csv', dtype=np.float, delimiter=",")
#         # Data[:, i] = np.concatenate(([midnum], np.concatenate(([codenum],OriData))))
#         Data[:, i] = np.concatenate(([midnum,codenum],OriData))
#         i=i+1
#         print midnum,'-',codenum
#
# np.savetxt('AutoEncoder/TraversingResult.csv', Data, delimiter = ',')
# print Data

# OriData = np.loadtxt('AutoEncoder/TraversingResult.csv', dtype=np.float, delimiter=",")
#
# Data=np.zeros((12,10,10))
#
# for i in range(12):
#     for j in range(len(OriData[:,0])):
#         midnum=int(OriData[j,0])
#         codenum=int(OriData[j,1])
#         indexmn = 11 - midnum
#         indexcn = 11 - codenum - 1
#         Data[i,indexcn,indexmn]=OriData[j,i+2]
#     np.savetxt('AutoEncoder/TraversingResult_'+str(i)+'.csv', Data[i,:,:], delimiter=',')



for i in range(12):
    data=np.loadtxt('AutoEncoder/TraversingResult.csv', dtype=np.float, delimiter=",")
    plt.figure('scatter midnum-codenum-avgER para_'+str(i+1))
    ax=plt.subplot(111,projection='3d')
    cm = plt.get_cmap("RdYlGn")
    col = cm(data[:,i+2] / (0.1))
    ax.scatter(data[:,0], data[:,1],data[:,i+2], c=col, marker='s')
    plt.title('para:'+str(i+1))
    ax.set_xlabel('midnum')
    ax.set_ylabel('codenum')
    ax.set_zlabel('avgER')
    plt.show()