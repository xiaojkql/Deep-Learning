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
    network_one = network.Network([784, 450, 10])


if __name__ == "__main__":
    main()
