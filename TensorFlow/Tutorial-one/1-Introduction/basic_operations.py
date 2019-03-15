# -*- coding: utf-8 -*-
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-15 22:29:24
'''


import tensorflow as tf

# 定义两个常数之间的操作
a = tf.constant([4, 5])
b = tf.constant([5, 6])

# 启动会话
with tf.Session() as sess:
    print("a = {},b = {}".format(sess.run(a), sess.run(b)))
    # element-wise
    print("Addition operations: {}".format(sess.run(a+b)))
    print("Multiplition operations: {}".format(sess.run(a*b)))
    c = tf.reshape(a, (1, 2))
    d = tf.reshape(b, (2, 1))
    mal = tf.matmul(c, d)
    print("Array multiplition: {}".format(sess.run(mal)))
