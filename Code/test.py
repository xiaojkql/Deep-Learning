import numpy as np
import os


def test1():
    print("first {}, second {}".format(1, 2))


def test2():
    print(np.zeros((4, 5)))  # 两个括号


def test3():
    a = np.random.randn(4, 5)  # 一个括号
    print(a.transpose())
    print(a.shape)


def test4():
    print(os.path.abspath('.'))


if __name__ == "__main__":
    test4()
