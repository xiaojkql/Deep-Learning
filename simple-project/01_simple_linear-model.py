#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 这个教程用一个简单的线性模型展示了使用Tensorflow的基本工作流程。在载入一个又手写数字的图像构成的额数字集后，我们用Tensorflow定义一个数学模型，并用tf来对其进行优化。然后对结果进行绘图分析与讨论。
#
# 预备知识:
# 熟悉基本的线性代数、pthon、jupyter notebook编辑器。这个教程也会帮助你理解基本的机器学习和分类。

# # Import


from sklearn.metrics import confusion_matrix
from mnist import MNIST
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# # Load data
data = MNIST(data_dir="data/MNIST")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validaton-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

image_size_flag = data.img_size_flat
img_shape = data.img_shape
num_classes = data.num_classes

data.y_test[:5]  # 单个会降维，slice不会降维
data.y_test_cls[:5]


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9  # 当不是9幅图像时候就会报错
    # 创建一个3X3的绘图网格
    fig, axes = plt.subplots(3, 3)  # fig表示整个图像的一个对象，axes表示每一个子图
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {},Pred: {}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        # 去除整个图像的x,y轴标签
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


images = data.x_test[:9]
cls_true = data.y_test_cls[:9]
plot_images(images=images, cls_true=cls_true)


x = tf.placeholder(tf.float32, [None, image_size_flag])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])  # 当不指明数组时，表示的就是单个的变量
weigths = tf.Variable(tf.zeros([image_size_flag, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))
logits = tf.matmul(x, weigths)+biases
y_pred = tf.nn.softmax(logits)  # 输出是啥呢？[0,1,0,0,0,0,0,0,0]
y_pred_cls = tf.argmax(y_pred, 1)  # 输出是啥呢？[1,2,3,0,4]就是预测输出
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
# [1,0,1,1,0,1,0,1,1,1,] -> N个
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

batch_size = 100


def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch, _ = data.random_batch(
            batch_size=batch_size)  # 产生批量数据
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)


feed_dict_test = {x: data.x_test,
                  y_true: data.y_test,
                  y_true_cls: data.y_test_cls}


def print_accuracy():
    # 运行计算精度accuracy的，喂入数据
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))


def print_confusion_matrix():
    cls_true = data.y_test_cls
    # cls_pred 运行计算run
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')


# 绘制错误分类的图片
def plot_example_errors():
    correct, cls_pred = session.run(
        [correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)
    error_images = data.x_test[incorrect]
    error_cls = cls_pred[incorrect]
    true_cls = data.y_test_cls[incorrect]
    plot_images(images=error_images[:9],
                cls_true=true_cls[:9],
                cls_pred=error_cls[:9])


def plot_weights():
    w = session.run(weigths)
    w_min = np.min(w)
    w_max = np.max(w)
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            image = w[:, i].reshape(img_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


print_accuracy()


plot_example_errors()


optimize(num_iterations=1)


print_accuracy()


plot_example_errors()


plot_weights()


optimize(num_iterations=9)


print_accuracy()


plot_example_errors()


plot_weights()


optimize(num_iterations=100)


print_accuracy()


plot_example_errors()


plot_weights()


print_confusion_matrix()
