# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-17 20:35:42
'''

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# dataSets
train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                         2.167, 7.042, 10.791,
                         5.313, 7.997, 5.654, 9.27, 3.1]).reshape(1, -1)
train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596,
                         2.53, 1.221, 2.827, 3.465,
                         1.65, 2.904, 2.42, 2.94, 1.3])
test_X = numpy.asarray(
    [6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1]).reshape(1, -1)
test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

n_samples = train_X.shape[1]
# 模型的输入数据
X = tf.placeholder(tf.float32, [1, None])
y = tf.placeholder(tf.float32, [None])  # 用于优化

# 模型的变量
W = tf.Variable(tf.random_normal([1, 1]))
b = tf.Variable(tf.random_normal([1]))

# build model
y_pred = tf.add(tf.matmul(W, X), b)
cost = tf.reduce_mean(tf.pow(y_pred-y, 2))/2
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(cost)
init = tf.global_variables_initializer()

# training parameters
iterations_num = 1000
epoch_step = 40

with tf.Session() as sess:
    sess.run(init)
    feed_dict_train = {X: train_X, y: train_Y}
    epoch_num = iterations_num/epoch_step
    for epoch in range(iterations_num):
        sess.run(optimizer, feed_dict=feed_dict_train)
        if (epoch) % epoch_step == 0:
            train_cost = sess.run(cost, feed_dict=feed_dict_train)
            epochth = epoch/epoch_step
            print("epochs: {0}/{1},Training Cost:{2}, W={3}, b={4}".format(
                epochth, epoch_num, train_cost, sess.run(W), sess.run(b)))
    train_cost = sess.run(cost, feed_dict=feed_dict_train)
    print("Training Finished!")
    print("Training cost: {}, W={}, b={}".format(
        train_cost, sess.run(W), sess.run(b)))

    feed_dict_test = {X: test_X, y: test_Y}
    test_cost = sess.run(cost, feed_dict=feed_dict_test)
    print("Testing cost: {}".format(test_cost))

    plt.plot(train_X.ravel(), train_Y, 'ro', label="Original data")
    plt.plot(train_X.ravel(), (sess.run(W)*train_X +
                               sess.run(b)).ravel(), label="Ftting Line")
    plt.legend()
    plt.show()
