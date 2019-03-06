# -*- coding: utf-8 -*-
import numpy as np
import random
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-06 01:00:55
'''


class Network:

    def __init__(self, sizes):
        """以sizes=[n,m,l]
        并初始化weights,与bias"""
        self.layer = len(sizes)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.bias = [np.random.randn(x, 1) for x in sizes[1:]]

    def feed_forward(self, a):
        """输入一个向量a以正向传播的方式计算最终的值"""
        for w, b in zip(self.weights, self.bias):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, trainging_data, epochs, minibatch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n_training = len(trainging_data)
        random.shuffle(trainging_data)  # 对训练数据集进行原地重新洗牌
        # 划分数据集成minibatch
        minibatchs = [trainging_data[k:k+minibatch_size]
                      for k in range(0, n_training, minibatch_size)]
        # 每一代外循环，每一个minibantchs进行内循环更新weights与bias
        for epoch in range(epochs):
            for minibatch in minibatchs:
                self.update_parameter(minibatch, eta)
            if test_data:
                accuracy = self.evaluate(test_data)
                print("epoch: {}, accuracy: {}".format(epoch, accuracy))

    def update_parameter(self, minibatch, eta):
        # 用于存储weights与bias关于此批训练数据集的偏导数
        nlpha_w = np.array([np.zeros(w.shape) for w in self.weights])
        nlpha_b = np.array([np.zeros(b.shape) for b in self.bias])
        # minibatch --> [(x,y),...]
        for x, y in minibatch:
            # 正向传播计算 a z
            # print(x)
            a_arr, z_arr = [x], []
            self.forward_pop(x, a_arr, z_arr)
            # 计算最后一层的误差
            delta_arr = [0]*(self.layer-1)
            cost_derivative = self.cost_derivative(a_arr[-1], y)
            delta_arr[-1] = cost_derivative*self.sigmoid_derivative(z_arr[-1])
            # print(delta_arr[-1]) (10,1)满足要求
            # 反向传播计算误差
            self.back_pop(delta_arr, z_arr)
            # 计算每一个样本的偏导数
            for i in range(self.layer-1):
                nlpha_w[i] += np.dot(delta_arr[i], a_arr[i].transpose())
                # print(nlpha_b[i])
                nlpha_b[i] += delta_arr[i]
        self.weights -= float(eta/len(minibatch))*nlpha_w
        self.bias -= float(eta/len(minibatch))*nlpha_b

    def back_pop(self, delta_arr, z_arr):
        for i in range(self.layer-3, -1, -1):
            delta = (np.dot(self.weights[i+1].transpose(), delta_arr[i+1]) *
                     self.sigmoid_derivative(z_arr[i]))
            delta_arr[i] = delta

    def forward_pop(self, x, a_arr, z_arr):
        # 正向传播计算a和z
        a = x
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, a)+b
            a = self.sigmoid(z)
            z_arr.append(z)
            a_arr.append(a)

    def evaluate(self, test_data):
        """输入test_data=[(x,y)]
        输出 test_data的精度"""
        n_test, accuracy = len(test_data), 0
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for x, y in test_data]
        return sum([int(x == y) for x, y in test_results])/n_test
        i

    def sigmoid(self, z):
        """激活函数"""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    def cost_derivative(self, a, y):
        return (a-y)
