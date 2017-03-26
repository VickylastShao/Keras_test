# coding: utf-8
# 机器学习的课程中，XOR问题是必讲的一个例题，它能很好地帮助理解神经网络的工作过程。
# 可是它只有4个样本，每个样本二维输入一维输出，太简单。实现来讲，体现不出神经网络的强大来。

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt


# input data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_label = np.array([0, 1, 1, 0])
# create models, with 1hidden layers
model = Sequential()#初始化sequential模型
model.add(Dense(32, init='uniform', input_dim=2))
#添加第一层 全连接层，32为输出维度，说明全连接层有32个节点，
# 权重初始化方法init定义为uniform，由于是第一层，所以要固定输入维度，为2维输入向量，即一组训练数据是2个参数
# 这里全连接层没有定义激活函数，等效于使用了线性激活函数，a(x)=x
model.add(Activation('relu'))
#添加激活层，要使用的激活函数是relu函数Rectified Linear Units 线性修正单元
model.add(Dense(1))
model.add(Activation('sigmoid'))
#添加第二层 全连接层，该层的输出维度为1，因为已经是最后一层了，而模型输出就是1
# 不用设置输入维度，因为上一层已经设置输出维度为32，所以这一层的输入维度一定是32
#添加激活层，要使用的激活函数是sigmoid，常用的连续非线性阈值函数
# training
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
# 训练过程配置 损失函数(也称目标函数)选择binary_crossentropy 亦称作对数损失，logloss
# sgd为随机梯度下降法(Stochastic gradient descent)
hist = model.fit(x_train, y_label, batch_size=1, nb_epoch=100, shuffle=True, verbose=0, validation_split=0.0)
# 模型训练 输入训练输入数据(x_train)和输出数据(y_label)
# batch_size：整数，指定进行梯度下降时每个batch包含的样本数,训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
# 本例中是1，就是每轮更新权值进行优化的时候只取一个样本计算其梯度，按照这个梯度进行优化
# 较小的batch_size会提高计算速度，但是会增大迭代次数，可能会发散，较大的batch_size会充分利用内存，减少迭代次数，但是每次计算速度慢
# nb_epoch：迭代训练的次数，100次后停止迭代
# shuffle：表示是否在训练过程中每个epoch前随机打乱输入样本的顺序
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等
print(hist.history)
# evaluating model performance
loss_metrics = model.evaluate(x_train, y_label, batch_size=1)
plt.plot(range((len(hist.history.get('acc')))),hist.history.get('loss'))
plt.show()

model.save('my_model.h5')
#save a Keras model into a single HDF5 file
