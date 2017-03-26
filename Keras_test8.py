# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.visualize_util import plot
from Keras_Defs import *
from keras.models import load_model
from matplotlib import animation


model=load_model('H5_models/DH_fan_model_0to150000_7L.h5')
preStart=150000
preNum=2000
data=np.load('data.npy')
ip=[0,1,2,3,4,5,6]      #输入参数index
op=[7,8,9,10,11,12,13,14,15,16,17,18]   #输出参数index
input = data[0:323000, ip]
output = data[0:323000, op]
testoutput,testresultReal,errAb,errRe=preDict(model,data,input,output,ip,op,preStart,preNum)

# plt.figure(1)
# plt.plot(testresultReal[:,1])
# plt.plot(output[preStart:preStart+preNum,1])
# plt.title("modeloutputvs realoutput 1")
# plt.show()

realoutput=data[preStart:preStart+preNum, op]

showoutput=0

fig,ax=plt.subplots()
x=np.arange(0,1000)
line1,=ax.plot(x,testresultReal[x,showoutput],label='Predict')
line2,=ax.plot(x,realoutput[x,showoutput],label='Real')

plt.legend(handles=[line1,line2],loc='best')


def animate(i):
    line1.set_ydata(testresultReal[x+i,showoutput])
    line2.set_ydata(realoutput[x + i, showoutput])
    return line1,line2,

def init():
    line1.set_ydata(testresultReal[x,showoutput])
    line2.set_ydata(realoutput[x, showoutput])
    return line1,line2,

ani=animation.FuncAnimation(fig=fig,func=animate,frames=500,init_func=init,interval=20,blit=False)
# ani.save('line.gif', dpi=80, writer='imagemagick')
ani.save('preDict.gif', writer='imagemagick', fps=10, dpi=100)
plt.show()
