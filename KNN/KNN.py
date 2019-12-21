"""
@File    : KNN.py
@Time    : 2019-10-22 12:27
@Author  : Lee
@Software: PyCharm
@Email   : leehaoran@pku.edu.cn
"""

# mnist_train:60000
# mnist_test:500

# acc: 0.956
# time: 1032s



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


    # 将数据转化为矩阵（matrix）
    x_label=np.mat(data[:,1:])

    #数据归一化，返回数据
    # y_label为np.ndarray,x_label为np.matrix
    return x_label/255,y_label


#计算两个点的距离
# 使用欧式距离 dis=sqrt(square（x1-x2）)
# x1，x2均为numpy.matrix ,1x784 行向量
def caculDistance(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2)))


# X_train为训练集中的数据，y_label为对应的分类标签
# x为所给的带分类数据
# k是我们判断最近k个点的k值
def findCluster(X_train,y_train,x,k):
    distances=[] #用于保存距离
    for x_train in X_train:
        #计算样本与训练集样本的距离
        dis=caculDistance(x_train,x)
        distances.append(dis)
    # 遍历完所有训练集数据之后,我们就可以进行统计分类
    # 可以知道distances的索引与y_label的索引一一对应
    # 如distances[1]=7.43，y_label[1]=1
    # 则x与分类结果 1 的距离为7.43

    #下面我们需要找到distances中最小的k个元素，并且返回他们的索引就可以了
    # argsort函数返回的是数组值从小到大的索引值
    # 将distances变为np.array，然后对索引排序，取前k个
    minK=np.argsort(np.array(distances))[:k]

    # 将对应分类结果添加到result中
    result=[]
    for i in range(k):
        result.append(y_train[minK[i]])

    #a=[1,2,3,4,4,4,4,4,4,4,1,1,1,23,2,2,2,3,1]
    # b=Counter(a)
    # b.most_common(1)
    # out:[(4,7)] 4是元素，7是出现次数
    #下面我们就可以在result中找出出现最多的值作为分类结果了

    belonging=Counter(result).most_common(1)[0][0]

    return belonging

def test(X_train,y_train,X_test,y_test,k):
    # 只测试500个

    acc_num=0# 正确个数
    acc=0 #正确率
    for i in range(500):
        cluster=findCluster(X_train,y_train,X_test[i],k)
        if cluster==y_test[i]:
            acc_num+=1
        print(f'find {i}th data cluster:cluster_pred={cluster},cluster={y_test[i]}')
        print('now_acc=',acc_num/(i+1))





if __name__=='__main__':
    # 获取当前时间
    start = time.time()

    #读取训练文件
    X_train,y_train=loadData('../Mnist/mnist_train.csv')

    # 读取测试文件
    X_test,y_test=loadData('../Mnist/mnist_test.csv')

    test(X_train,y_train,X_test,y_test,k=20)

    #获取结束时间
    end=time.time()

    print('run time:',end-start)





























