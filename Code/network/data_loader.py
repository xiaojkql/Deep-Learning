# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-06 11:03:34
'''

import pickle
import gzip
import numpy as np
import os


def load_data():
    f = gzip.open(
        r"D:\MyGithub\Deep-Learning\Code\data\mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data


def data():
    # data = [[inputs],[results]]
    tr_data, v_data, te_data = load_data()
    tr_inputs = [np.reshape(x, (784, 1)) for x in tr_data[0]]
    tr_results = [vectorize_result(y) for y in tr_data[1]]
    training_data = list(zip(tr_inputs, tr_results))
    v_inputs = [np.reshape(x, (784, 1)) for x in v_data[0]]
    validation_data = zip(v_inputs, v_data[1])
    te_inputs = [np.reshape(x, (784, 1)) for x in te_data[0]]
    test_data = list(zip(te_inputs, te_data[1]))
    return training_data, validation_data, test_data


def vectorize_result(j):
    # 向量化数据 [0,0,1,0,0,0,0,0,0]
    e = np.zeros((10, 1))  # 所用np来操作矩阵 二维列向量
    e[j] = 1
    return e


if __name__ == "__main__":
    training_data, validation_data, test_data = data()
    print(list(training_data)[1])
