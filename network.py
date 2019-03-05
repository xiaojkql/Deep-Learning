# -*- coding: utf-8 -*-
import numpy as np
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-06 01:00:55
'''


class Network:

    def __init__(self, sizes):
        """以sizes=[n,m,l]
        并初始化weights,与bias"""
        n = len(sizes)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.bias = [np.random.randn(x) for x in sizes[1:]]

    def sigmoid(self, z):
        """激活函数"""
        return 1/(1+np.exp(-z))

    def feed_forward(self, a):
        """输入一个向量a以正向传播的方式计算最终的值"""
        for w, b in zip(self.weights, self.bias):
            a = self.sigmoid(np.dot(w, a)+b)
        return a
