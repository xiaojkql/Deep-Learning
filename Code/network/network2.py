# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-07 10:25:02
'''

# 辅助类与函数
# 两个代价函数的类
# S型函数的原型以及它的偏导数形式
import numpy as np
import random


# S型函数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# S型函数的偏导数
def sigmoid_derivative(z):
    return sigmoid(z)*(1.0-sigmoid(z))


class Quadratic_cost:

    # 计算代价 --> 返回的是一个值，即对于一个输入的代价值
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    # 计算最后一层的delta 是化简以后与代价函数的关系
    @staticmethod
    def delta(a, z, y):
        return (a-y)*sigmoid_derivative(z)


class Entropy_cross_cost:

    # 计算代价 --> 计算的是一个值，即对于一个输入的代价值
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-(y*np.log(a)+(1-y)*np.log(1-a))))

    # 计算最后一层的delta，即关于代价函数C化简后的形式
    @staticmethod
    def delta(a, z, y):
        return (a-y)


# 下面是神经网络的类了
# 使用方法--> 首先初始化该类，然后调用SGD进行该网络的优化，寻找好的网络的参数
class Network:

    # 初始化函数
    def __init__(self, sizes, cost=Entropy_cross_cost):
        """
        设置参数，并初始化网络的权重以及偏置项
        sizes --> 网络层信息
        cost --> 该网络使用的代价函数
        """
        self.lays_num = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_initial_w_b()  # 初始化权重与偏置项

    # 默认的初始化方式，即修改方差的初始化方法
    def default_initial_w_b(self):
        self.weights = np.array([np.random.randn(y, x)/np.sqrt(x) for x, y in
                                 zip(self.sizes[:-1], self.sizes[1:])])
        self.bias = np.array([np.random.randn(x, 1) for x in self.sizes[1:]])

    # 未修改的初始化方法
    def origin_initial_w_b(self):
        self.weights = [np.random.randn(y, x) for x, y in zip(
            self.sizes[:-1], self.sizes[1:])]
        self.bias = [np.random.randn(x, 1) for x in self.size[1:]]

    # 优化权重，偏置项的总入口
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_traing_accuracy=False):
        n_training = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n_training, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, n_training)  # 用每一个小批量进行更新数据
            if monitor_training_cost:
                training_cost.append(self.cal_cost(training_data))
            if monitor_traing_accuracy:
                training_accuracy.append(self.accuracy(training_data))
            if evaluation_data:
                if monitor_evaluation_accuracy:
                    evaluation_accuracy.append(self.accuracy(evaluation_data))
                if monitor_evaluation_cost:
                    evaluation_cost.append(self.cal_cost(evaluation_data))
                print("epoch {}, accuracy {}".format(
                    epoch, evaluation_accuracy[-1]/len(evaluation_data)))
        return (evaluation_cost, evaluation_accuracy,
                training_cost, training_accuracy)

    def cal_cost(self, data):
        cost = 0
        for x, y in data:
            cost += self.cost.fn(self.feed_forward(x), y)
        cost_temp = [self.cost.fn(self.feed_forward(x), y) for x, y in data]
        return sum([self.cost.fn(self.feed_forward(x), y) for x, y in data])

    def accuracy(self, data):
        results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                   for x, y in data]
        return np.sum([int(x == y) for x, y in results])

    def feed_forward(self, x):
        for w, b in zip(self.weights, self.bias):
            x = sigmoid(np.dot(w, x) + b)
        return x

    # 使用小批量随机更新权重与偏置

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nlpha_w = np.array([np.zeros(w.shape) for w in self.weights])
        nlpha_b = np.array([np.zeros(b.shape) for b in self.bias])
        for x, y in mini_batch:
            # feedpop
            activations, z_arr = [x], []
            for w, b in zip(self.weights, self.bias):
                z = np.dot(w, x)+b
                x = sigmoid(z)
                activations.append(x)
                z_arr.append(z)
            delta_arr = [0]*(self.lays_num-1)
            delta_arr[-1] = self.cost.delta(activations[-1], z_arr[-1], y)
            # backpop
            for i in range(self.lays_num-3, -1, -1):
                delta = np.dot(self.weights[i+1].transpose(),
                               delta_arr[i+1]) * sigmoid_derivative(z_arr[i])
                delta_arr[i] = delta  # delta是numpy对象
            # 累加delta
            for i in range(self.lays_num-1):
                nlpha_w[i] += np.dot(delta_arr[i], activations[i].transpose())
                nlpha_b[i] += delta_arr[i]
        # 更新w与b
        self.weights = self.weights - eta / \
            len(mini_batch)*(nlpha_w) - eta*lmbda/n*self.weights
        self.bias -= eta/len(mini_batch)*nlpha_b
