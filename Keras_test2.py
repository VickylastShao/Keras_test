# coding: utf-8
# 绘制3D曲面的例子

from keras.models import load_model
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
model=load_model('my_model.h5')

# test_x=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# arr = []
# for i in range (10):
#     x=0.1*i
#     for j in range(10):
#         y = 0.1 * j
#         arr.append([x, y])
#
# test_x=np.array(arr)
#
# result=model.predict(test_x,batch_size=1)

X=np.arange(-1,2,0.1)
Y=np.arange(-1,2,0.1)
X,Y=np.meshgrid(X,Y)
arr=np.zeros((len(X)*len(Y),2))
m=0
for i in range (len(X)):
    for j in range(len(Y)):
        arr[m] = np.hstack((X[i][j], Y[i][j]))
        m+=1
result = model.predict(arr, batch_size=1)
Z=np.zeros((len(X),len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i][j] = result[i*len(Y)+j]

fig=plt.figure()
ax=Axes3D(fig)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
plt.show()
print("end")

