"""
@File    : perceptron_Mnist.py
@Time    : 2019-10-18 11:55
@Author  : 李浩然
@Software: PyCharm
"""

import pandas as pd
import numpy as np
import time

# mnist_train:60000
# minst_test:10000
# acc:79.04%
# time:185s


def loadData(fileName):
    #从文件中读取数据
    data=pd.read_csv(fileName,header=None)
    # 将数据从dataframe转化为np.array
    data=data.values
    #数据第一行为分类结果
    y_label=data[:,0]
    #感知机解决的是2分类的问题，所以我们将<5 >=5分为两类
    y_label[y_label<5]=-1
    y_label[y_label>=5]=1
    # 将数据转化为矩阵（matrix）
    x_label=np.mat(data[:,1:])

    #数据归一化，返回数据
    return np.mat(x_label/255),np.mat(y_label).T


#标准化
#Z-score标准化方法
#这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
#经过处理的数据符合标准正态分布，即均值为0，标准差为1，转化函数为：
# def norm(x_data):
#
#     #均值
#     mu=np.zeros((1,x_data.shape[1]))
#
#     # 标准差
#     sigma=np.zeros((1,x_data.shape[1]))
#     #计算均值与标准差
#     # axis=0代表计算每一列的平均值
#     # axis=1代表计算每一行的均值
#     # 默认计算所有值
#     mu=x_data.mean(axis=0)
#     sigma=x_data.std(axis=0)
#     # 归一化
#     x_norm=(x_data-mu)/(sigma)
#     return x_norm


def perceptron(x_data, y_label):
    # y=w*x+b

    # #将data转置
    # x_data=x_data.T

    # 初始化w为全0，长度与每一个样本特征一致
    w = np.zeros((x_data.shape[1], 1))  # 784x1的列向量
    # 初始b=0
    b = 0
    # 学习率  也就是我们梯度下降的步长
    alpha = 0.001
    # 迭代次数
    iters = 100

    m, n = x_data.shape  # 返回x_data的维度（行，列）
    for iter in range(iters):
        # 计算每个循环错误样本数，计算正确率
        count = 0

        for i, x in enumerate(x_data):
            # 计算需要把握好每一个向量的维度，才不容易出错
            # x_data是60000x784，x是1x784，y_label是784x1
            # x是一个样本，y是样本正确分类的结果
            y = y_label[i]

            #在感知机中，误分类点 y_(w*x+b)与y异号，相乘小于0
            # 等于0则点落在超平面上，也不符合要求
            if y * (x @ w + b) <= 0:
                # 这些点被归于误差点

                # 根据公式进行梯度下降，使用的是随机梯度下降
                # 每遍历一个误分类样本点，就根据当前样本点进行梯度下降更新参数
                # x是1x784行向量，w是784x1的列向量
                # 所以梯度下降更新w时候，x需要转置
                # y是1x1的矩阵,x.T是784x1的矩阵,取其中的值相乘用np.multiply()
                w += alpha * np.multiply(y, x.T)
                b += alpha * y
                # 误分类样本加一
                count += 1
        # 计算分类正确率
        print(count, m)
        acc = (m - count) / m
        print('Round %d' % (iter), end=' ')
        print('acc=', acc)

    # 返回训练完毕的w，b
    return w, b

def test(x,y,w,b):
    # x：10000x784 矩阵
    # y：10000x1 列向量
    # 计算根据训练的参数得到的预测y值
    y_pred=x@w+b  # 10000x1 列向量
    m,n=y_pred.shape
    correct=0
    # f（x）=sign（wx+b）：计算值大于0就=1，小于0就等于-1
    y_pred[y_pred>=0]=int(1)
    y_pred[y_pred<0]=int(-1)

    #遍历计算结果是否正确
    for i in range(m):
        if y_pred[i]==y[i]:
            correct+=1
    acc=correct/m
    return  acc


if __name__=='__main__':
    # 获取当前时间
    start = time.time()

    #读取训练文件
    x_train,y_train=loadData('../Mnist/mnist_train.csv')
    # x_train_norm=norm(x_train)

    # 进行训练得到最优解w，b
    w,b=perceptron(x_train,y_train)
    # 读取测试数据
    x_test,y_test=loadData('../Mnist/mnist_test.csv')
    # x_test_norm=norm(x_test)

    #验证
    acc=test(x_test,y_test,w,b)
    print(acc)
    print('acc=',acc)

    #获取结束时间
    end=time.time()

    print('run time:',end-start)
