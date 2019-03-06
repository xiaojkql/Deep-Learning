# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-06 11:24:28
'''

from network import network
from network import data_loader


def main():
    # 载入数据
    training_data, validation_data, test_data = data_loader.data()
    network_one = network.Network([784, 100, 10])
    # trainging_data, epochs, minibatch_size, eta, test_data
    network_one.SGD(training_data, 10, 20, 1, test_data=test_data)


if __name__ == "__main__":
    main()
