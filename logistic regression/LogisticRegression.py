"""
@File    : LogisticRegression.py
@Time    : 2019-10-25 08:48
@Author  : Lee
@Software: PyCharm
@Email   : leehaoran@pku.edu.cn
"""


# mnist_train:60000
# mnist_test:500

# acc: 0.9913
# time:145s



import pandas as pd
import numpy as np
import time
import random



def loadData(fileName):

    data=pd.read_csv(fileName,header=None)

    # 增加一列常数项X0=1
    data[785]=1

    #将dataframe转化为numpy.array
    data=data.values

    #从样本中切分出分类结果
    y_label = data[:, 0]
    X_label = data[:, 1:]

    # 由于使用二分类逻辑回归
    # 将数据分为两类，y>5一类，y<5一类
    y_label[y_label>0]=1

    # 按照5分界，样本点很难找出一个较好的超平面区分开，正确率只有80多
    # y_label[y_label<5]=0
    # y_label[y_label>=5]=1

    return X_label,y_label
def sigmoid(X):
    # 定义sigmoid函数

    # 定义一个最小值，防止出现很接近0的数字，导致log无限大，loss为nan
    # minhx=np.exp(-3)
    # hx=1 / (1 + np.exp(-1 * X))
    # hx[hx<minhx]=minhx
    return 1 / (1 + np.exp(-1 * X))


def logisticRegression(X_train,y_train,epochs):
    # 随机生成参数w
    # 将其转化为列向量
    # w=np.mat([0. for _ in range(len(X_train[0]))]).reshape(-1,1)
    w=np.mat([random.uniform(0,1) for _ in range(len(X_train[0]))]).reshape(-1,1)
    # 将样本转化为矩阵
    X_train=np.mat(X_train)
    y_train=np.mat(y_train)

    # 进行训练
    print('start to train')

    learning_rate=0.001
    for i in range(epochs):
        # w是785x1，X_train是60000x785
        # hx为60000x1
        # hx>=0.5,y_predict=1;hx<0.5,y_predict=0
        hx=sigmoid(X_train@w)
        # 计算一下损失函数


        # loss=-1*(y_train@np.log(hx)+(1-y_train)@np.log(1-hx))
        print(f'in {i} epoch')

        # 下面对系数进行梯度上升梯度下降，从而降低损失函数
        # grad=X*(hx-y)
        # print(hx.shape) （60000，1）
        # print(y_train.shape)   （1，60000）
        # 60000x785 @ 60000x1=60000x1 与w维度匹配
        # 直接进行矩阵运行，写起来更加简洁，不用在写for循环遍历
        w-=learning_rate*X_train.T@(hx-y_train.T)

    # 返回w系数
    return w

def predict(x,w):
    # x为1x785 w：785x1
    # hx>=0.5,意味着逻辑回归认为改样本有超过50%的纪律为1，所以我们预测y_predict=1;
    # 同理，hx<0.5,y_predict=0
    hx=sigmoid(x@w)
    if hx>=0.5:
        return 1
    if hx<0.5:
        return 0


def test(X_test,y_test,w):
    acc=0 #正确率
    acc_num=0 #正确个数
    for i in range(len(X_test)):
        x=np.mat(X_test[i])
        y_pred=predict(x,w)
        if y_pred==y_test[i]:
            acc_num+=1
        print(f'find {i}th data cluster:y_pred={y_pred},y={y_test[i]}')
        print('now_acc=', acc_num / (i + 1))

if __name__=="__main__":
    # 获取当前时间
    start = time.time()

    # 读取训练文件
    print('load TrainData')
    X_train, y_train = loadData('../Mnist/mnist_train.csv')

    # 读取测试文件
    print('load TestData')
    X_test, y_test = loadData('../Mnist/mnist_test.csv')

    # 进行训练，得到系数
    # 循环200次
    w=logisticRegression(X_train,y_train,200)
    test(X_test, y_test,w)

    # 获取结束时间
    end = time.time()

    print('run time:', end - start)
































