引入新的初始化方法以及正则化技术以后的神经网络的编程架构实现

使用这个类的方法：
首先使用初始化创建类
然后调用SGD进行优化这个Network,并返回相应的训练结果，需要对训练进行评测时也是使用这个接口进行相应的输入参数设置

# 1 定义了代价函数函数的类
    类中包含两个函数方法
    包含：所用代价函数的原始形式
        代价函数的求导，关于Z---即输出层的delta

# 2 神经网络的类

## 2.1 初始化方法
        num_layers --> 神经网络的层数
        sizes --> 神经网络的架构设置
        调用初始化函数
        cost --> 选择代价函数，很巧妙，用这样一个类名来调用相应的代价函数，而类中提供统一的接口，函数形式，以及偏导数形式

## 2.2 默认的初始化方法
        即采用跟更新后的初始化方法

## 2.3 更新前的初始化方法
        即采用原来的 初始化方法

## 2.4 前向传播计算每一个输入的输出层结果
        进行预测，给输入，求输出

## 2.5 SGD --> 应用随机优化该神经网络的函数接口
        参数：
        trainin_data
        epochs
        mini_batch_size
        eta
        lambda
        evaluation_data
        monitor_evaluation_cost
        monitor_evaluation_accuracy
        monitor_traning_cost
        monitor_training_accuracy

## 2.6 update_mini_batch --> 应用小批量的数据更新网络参数
        参数：
        mini_batch
        eta
        lambda
        n

## 2.7 backpop(x,y) -->后向传播计算delta的接口

## 2.8 accuracy(data,convert=False)
        即计算精度

## 2.9 total_cost(data,lamda,convert=False)
        计算代价

## 2.10 save(filename)
        将神经网络的
        sizes
        weights
        bias
        cost类型用一个字典进行保存
        用了json模块

# 3 函数load
    从保存网络的文件中恢复一个已经训练好的网络对象

# 4 vector_result
    对结果进行向量化
    因为给的结果是指出为数字几，而我们的网络输出为一个向量

# 5 sigmoid函数
    给出S型函数的计算形式

# 6 sigmoid_derivative
    给出S型函数的偏导数的计算形式





