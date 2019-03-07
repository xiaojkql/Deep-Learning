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


def test5():
    ls = [1, 2, 3, 45, 6]
    print()


def test6():
    a = np.array([[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]])
    b = np.array([[1], [2], [3], [4], [5]])
    print(3.2*a)


def test7():
    # 当涉及到矩阵运算与广播时，尽量多用numpy,少用list
    # [1,2,3] * 2 = [2,4,6]
    print([1, 2, 3]*2)  # ---> [1, 2, 3, 1, 2, 3]
    print(np.array([1, 2, 3])*2)  # --->[2 4 6]


def test8():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a+np.array([1, 2, 3]))


def test9():
    a = np.array([np.zeros((4, 5)), np.ones((4, 5))])
    b = [sum(c) for c in a]  # 能直接用sum进行广播运算
    b = [c for x in a for c in x]  # 嵌套的循环在后面
    print(b)


def test10():
    a = np.array([1, 2, 3])
    print(1.0/a)  # 除法也同样是采用广播的方式


if __name__ == "__main__":
    test10()
