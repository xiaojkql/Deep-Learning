程序的组织结构

给出的原始数据：
TD:是由（x，y）这样的的数据结构给出的

将该神经网络进行封装成一个类
Class Network

类的方法
__init__(sizes) 以该神经网络的各层的神经元数量作为输入创建神经网络

feedforward(a) 计算输入为a时的神经网络的输出

SGD(TD,Epochs,minibatch_size,eta,test_data)

TD 训练集数据
epochs:训练的轮数
minibatch_size: 随机批的数量
随机训练的策略：
将样本集分成几个batch,然后用每个batch来更新权重与偏置

update_minibatch(minibatch,eta)
用方向传播来计算

backprop（x,y）反向传播计算关于权重与偏置的偏导数

evaluate(test_data)测试

cost_derivative(act,y)计算最后一层的误差

sigmoid(z)函数


sigmoid(z)的导数
