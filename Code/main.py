# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-06 11:24:28
'''

from network import network
from network import data_loader
from network import network_full_matrix
from network import network2


def main():
    # 载入数据
    training_data, validation_data, test_data = data_loader.data()
    network_one = network.Network([784, 50, 10])
    network_two = network2.Network([784, 100, 50, 10])
    # trainging_data, epochs, minibatch_size, eta, test_data
    # network_one.SGD(training_data, 10, 15, 1, test_data=test_data)
    network_two.SGD(training_data, 30, 15, 0.25, 5,
                    evaluation_data=test_data,
                    monitor_evaluation_accuracy=True)


if __name__ == "__main__":
    main()
