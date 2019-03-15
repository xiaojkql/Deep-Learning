# -*- coding: utf-8 -*-
import tensorflow as tf
'''
Author: Qin Yuan
E-mail: xiaojkql@163.com
Time: 2019-03-15 22:12:37
'''

"""
Tensorflow 会将一系列的op存放在一张graph中
一个graph也会存放一些数据
这些数据通过一系列的op相互连接
op会产生一些中间数据
而一些op的的数据来源于这些中间数据
所以在运行这些Op时同样也会运行它上游的op一层接一层

要想运行这些op必须通过会话窗口session
就好像是tf给我们提供的一个shell
"""


# 创建一个常数的op,此op被添加到default graph
# a 表示该op的输出值
# 将tf的很多东西看成一个op,其representions表示它们的输出
a = tf.constant("Hello World!")

# 开始tf的会话
with tf.Session() as sess:
    print(sess.run(a))
