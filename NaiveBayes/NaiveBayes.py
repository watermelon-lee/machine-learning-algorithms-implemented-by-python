"""
@File    : NaiveBayes.py
@Time    : 2019-10-22 16:24
@Author  : 李浩然
@Software: PyCharm
"""

# mnist_train:60000
# mnist_test:10000
# acc: 0.8427
# time: 129s


import pandas as pd
import numpy as np
import time
from collections import Counter




def loadData(fileName):
    #从文件中读取数据
    data=pd.read_csv(fileName,header=None)
    # 将数据从dataframe转化为ndarray
    data=data.values
    #数据第一行为分类结果
    y_label=data[:,0]
    x_label=data[:,1:]

    #数据二值化，返回数据
    #因为xi的取值范围为0-255，则计算p(X=xi\Y=y)的时候可能性过多，计算过于繁杂
    # 所以进行二值化
    # y_label为np.ndarray,x_label为np.ndarray

    x_label[x_label<128]=0
    x_label[x_label>=128]=1

    return x_label,y_label


#计算先验概率以及条件概率
def caculPrior(X_train,y_train):
    y_class=10 # 分类的10种结果
    feature_x=784 # x特征有784列

    Iy=np.array([0]*y_class)  # 出现次数
    Py=np.array([0]*y_class)  # P(Y=y)
    Ixy=np.zeros((y_class,feature_x,2)) #出现次数
    Pxy=np.zeros((y_class,feature_x,2))  # P(X=x|Y=y)

    # ！！！ 我们这里是有784个特征，784个0-1的数字相乘，会得到一个非常小的数
    # 很有可能造成下溢出。所以我们将其进行log处理
    # 可以知道log（x）随着x单增。px越大。

    #计算先验概率P(Y=y)
    for i in range(len(y_train)):
        # 先计算出现次数，之后在➗总数
        Iy[y_train[i]]+=1

    # P(Y=yi)=I(Y=yi)/N
    Py=np.log(Iy/len(y_train))

        # P(X=xi|Y=yi)=P(X=xi,Y=yi)/P(Y=yi)
    # 先计算出出现次数
    for i in range(len(X_train)):
        print(i)

        for j in range(len(X_train[i])):
            # Ixy=np.zeros((y_class,feature_x,2))
            # 所以Ixy是一个三位数组（10x784x2）

            Ixy[y_train[i]][j][X_train[i][j]]+=1

    print('***********************')
    # ➗总次数作为概率
    # P(X=xi|Y=yi)=I(Y=yi,X=xi)/I(Y=yi)
    for i in range(y_class):
        print(i)
        for j in range(feature_x):
            # 加上普拉斯平滑，避免出现0的情况
            # P(X=xi|Y=yi)=(I(Y=yi,X=xi)+lambda)/(I(Y=yi)+lambda*Sj)
            # 此处Sj=2，取值为0，1。lambda取1
            Pxy[i][j][0]=np.log((Ixy[i][j][0]+1)/(Iy[i]+2))
            Pxy[i][j][1]=np.log((Ixy[i][j][1]+1)/(Iy[i]+2))

    return Py,Pxy


def naiveBayes(Py,Pxy,x_test):
    y_class = 10  # 分类的10种结果
    feature_x = 784  # x特征有784列

    P=[0 for _ in range(y_class)]# 用来存放Y=0-9的可能性

    # x_test 1x784
    for i in range(y_class):
        for j in range(feature_x):
            # y=i的可能性
            P[i]+=Pxy[i][j][x_test[j]]
        # 记得要加上Py[i] ！！！
        P[i]=P[i]+Py[i]
    return P.index(max(P))

def test(X_train, y_train, X_test, y_test):
    Py,Pxy=caculPrior(X_train,y_train)

    acc_num=0
    acc=0
    for i in range(len(X_test)):
        y_pred=naiveBayes(Py,Pxy,X_test[i])
        if y_pred==y_test[i]:
            acc_num+=1
        print(f'find {i}th data cluster:y_pred={y_pred},y={y_test[i]}')
        print('now_acc=', acc_num / (i + 1))

if __name__=="__main__":
    # 获取当前时间
    start = time.time()

    # 读取训练文件
    X_train, y_train = loadData('../Mnist/mnist_train.csv')

    # 读取测试文件
    X_test, y_test = loadData('../Mnist/mnist_test.csv')

    test(X_train, y_train, X_test, y_test,)

    # 获取结束时间
    end = time.time()

    print('run time:', end - start)

