"""
@File    : Perceptron.py
@Time    : 2019-10-15 13:14
@Author  : 李浩然
@Software: PyCharm
"""

import pandas as pd
import numpy as np


def loadData(fileName):
    data_train = pd.read_csv(fileName)



    # titanic训练集所有特征如下：
    # passengerId 乘客编号
    # survived 是否存活 1是 0否
    # pclass 船舱等级  1=lst 2=2nd 3=3rd
    # name 姓名
    # sex 性别
    # age 年纪
    # sibsp 🚢上的兄弟姐妹/配偶个数
    # parch 🚢上的父母，孩子
    # ticket 船票号码
    # fare 船票价格
    # cabin 船仓号
    # embarked 登船港口  C = Cherbourg, Q = Queenstown, S = Southampton

    # Cabin船舱号有大量空值，对于空值填充可能有较大误差，所以我们先不考虑cabin作为特征
    # age，由于age缺失很少，我们使用年龄的平均值进行填充
    # passengerId是一个连续的序列，与结果无关，我们不选择这个作为特征
    # ticket是船票序列，我们不分析
    # embarked和sex这两个特征是字符串，进行处理
    # 将sex中male=1，famle=0
    # embarked中 c=1，q=2，s=3

    # 我们选取其中可以转化为数字特征的特征
    feature = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", ]

    # 下面处理一些缺失值，进行填充.然后将一些特征转化为数字

    # 使用平均值填充Age
    data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

    ## embarked中有两条缺失的，我们使用其中出现最多的来填充
    ans = data_train['Embarked'].value_counts()

    # 返回最大值索引
    fillstr = ans.idxmax()
    # 填充
    data_train['Embarked'] = data_train['Embarked'].fillna(fillstr)

    # 我们发现embarked和sex这两个特征是字符串，进行处理
    # 我们将sex中male=1，famle=0
    # 将embarked中 c=1，q=2，s=3

    # data_train['Sex'][data_train['Sex']=='male']=1
    # data_train['Sex'][data_train['Sex']=='female']=0

    # 使用loc定位行列
    # data_train["Sex"] == "male"定位行 sex是列
    data_train.loc[data_train["Sex"] == "male", "Sex"] = 0
    data_train.loc[data_train["Sex"] == "female", "Sex"] = 1
    data_train.loc[data_train['Embarked'] == 'C', 'Embarked'] = 0
    data_train.loc[data_train['Embarked'] == 'Q', 'Embarked'] = 1
    data_train.loc[data_train['Embarked'] == 'S', 'Embarked'] = 2

    #选取出我们需要的特征作为训练集
    x_train = data_train[feature]
    #将dataframe转为矩阵
    x_train=np.mat(x_train)

    # 分类结果label
    y_label = data_train['Survived']
    # 给我们的y分类是0，1，而在感知机中，分类为-1，1
    # 我们将其中的死亡0，转变为死亡-1
    y_label.loc[y_label==0]=-1
    # 将dataframe转为列向量 891x1
    y_label=np.mat(y_label).T
    return x_train,y_label

#标准化
#Z-score标准化方法
#这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
#经过处理的数据符合标准正态分布，即均值为0，标准差为1，转化函数为：
def norm(x_data):

    #均值
    mu=np.zeros((1,x_data.shape[1]))

    # 标准差
    sigma=np.zeros((1,x_data.shape[1]))
    #计算均值与标准差
    # axis=0代表计算每一列的平均值
    # axis=1代表计算每一行的均值
    # 默认计算所有值
    mu=x_data.mean(axis=0)
    sigma=x_data.std(axis=0)
    # 归一化
    x_norm=(x_data-mu)/(sigma)
    return x_norm


def preceptron(x_data, y_label):
    # y=w*x+b

    # #将data转置
    # x_data=x_data.T

    # 初始化w为全0，长度与每一个样本特征一致
    w = np.zeros((x_data.shape[1], 1))  # 7x1的列向量
    # 初始b=0
    b = 0
    # 学习率  也就是我们梯度下降的步长
    alpha = 0.0001
    # 迭代次数

    iters = 500

    m, n = x_data.shape  # 返回x_data的维度

    for iter in range(iters):
        # 计算每个循环错误样本数，计算正确率
        count = 0

        for i, x in enumerate(x_data):
            # 计算需要把握好每一个向量的维度，才不容易出错
            # x_data是891x7，x是1x7，y_label是891x1

            y = y_label[i]

            if y * (x @ w + b) <= 0:
                # 这些点被归于误差点

                # x是1x7行向量，w是7x1的列向量
                # 所以梯度下降更新w时候，x需要转置
                # y是1x1的矩阵,x.T是7x1的矩阵,取其中的值相乘用np.multiply()
                w += alpha * np.multiply(y, x.T)
                b += alpha * y
                # 误分类样本加一
                count += 1
        # 计算分类正确率
        print(count, m)
        acc = (m - count) / m
        print('Round %d' % (iter), end=' ')
        print('acc=', acc)
        print(w, b)

    # 返回训练完毕的w，b
    return w, b



def test(fileName,w,b):
    # 对data——test进行一样的预处理
    # 填充缺失值，将字符特征转化为数字特征
    data_test = pd.read_csv(fileName)
    ans = data_test['Embarked'].value_counts()
    # 返回最大值索引
    fillstr = ans.idxmax()
    data_test['Embarked'] = data_test['Embarked'].fillna(fillstr)
    data_test.info()
    data_test.loc[data_test["Sex"] == "male", "Sex"] = 0
    data_test.loc[data_test["Sex"] == "female", "Sex"] = 1
    data_test.loc[data_test['Embarked'] == 'C', 'Embarked'] = 0
    data_test.loc[data_test['Embarked'] == 'Q', 'Embarked'] = 1
    data_test.loc[data_test['Embarked'] == 'S', 'Embarked'] = 2
    data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
    mid = data_test['Fare'].median()
    data_test['Fare'] = data_test['Fare'].fillna(value=mid)


    # 选取的特征值
    feature = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", ]
    x_test=np.mat(data_test[feature])

    #计算结果 y=wx+b
    test_predictions = x_test@w+b
    # 因为结果是二分类，大于0的我们归类到1，小于0归类到-1（但是本题目中为0）
    test_predictions[test_predictions >= 0] = 1
    test_predictions[test_predictions < 0] = 0 # 本题目中0就是-1。0代表死亡
    #将结果变为列向量
    test_predictions=np.array(test_predictions.T)
    # 结果变为一位数组
    test_predictions=test_predictions.flatten()


    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':test_predictions.T.astype(np.int32)})
    result.to_csv('my_submission.csv',index=False)

if __name__=='__main__':
    x_train,y_label=loadData('Titanic_data/train.csv')
    x_norm=norm(x_train)
    w,b=preceptron(x_norm,y_label)
    test('Titanic_data/test.csv',w,b)


# 正确率67%