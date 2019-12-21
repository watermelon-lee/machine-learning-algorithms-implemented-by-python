"""
@File    : Adaboost(Boost_Tree).py
@Time    : 2019-12-04 15:00
@Author  : Lee
@Software: PyCharm
@Email: leehaoran@pku.edu.cn
"""


# mnist_train:60000
# mnist_test:10000


# 实际训练使用 1000
# 实际预测使用 200
# acc: 0.98
# time: 285s



import pandas as pd
import numpy as np
import time


def loadData(fileName):
    #从文件中读取数据
    data=pd.read_csv(fileName,header=None)
    # 将数据从dataframe转化为ndarray
    data=data.values
    #数据第一行为分类结果
    y_label=data[:,0]
    x_label=data[:,1:]

    #数据二值化，返回数据
    #因为xi的取值范围为0-255，那么划分点太多了，我们进行二值化
    # 二值化之后，我们使用-0.5,0.5,1.5三个点即可
    x_label[x_label<128]=0
    x_label[x_label>=128]=1

    # 以5作为分界下效果不好，正确率在80左右。也明显强于了50%
    # y_label[y_label<5]=-1
    # y_label[y_label>=5]=1

    # 以0作为分类。0设置为-1，其他设置为1
    y_label[y_label ==0 ] = -1
    y_label[y_label>=1]=1

    # np.ndarray
    return x_label,y_label


def cal_Gx_e(X,Y,div,rule,D,feature):
    '''
    用于计算在该特征下，使用条件为rule，样本权重分布为D，划分点为div，返回划分结果和误差率
    :param:X 样本
    :param:Y 标签
    :param div: 划分点
    :param rule: 划分规则，大于div为1还是0
    :param D: 样本权重分布
    :param feature: 样本的的几个特征（总共有784（28*28）个）
    :return: Gx，e
    '''

    x=X[:,feature] #拿出我们选择的一列

    # rule分为LessIsOne：即小于划分点为1，大于为0。BiggerIsOne：大于划分点为1，小于为-1
    Gx=[]
    e=0
    if rule=='LessIsOne':
        L,B=1,-1
    else:
        L,B=-1,1
    for i in range(len(x)):
        #依据样本点在该划分点的左右，预测GX
        if x[i]>div: #右侧，预测为B，即大于划分点
            Gxi=B
            Gx.append(Gxi)
        else:
            Gxi=L
            Gx.append(Gxi)
        if Gxi!=Y[i]:
            e+=D[i] #错误分类， 根据公式8.1累加计算e

    # 之后要进行计算下一轮的权值，使用np.array方便于使用向量的方式一次计算所有权值
    return np.array(Gx),e

def create_single_boosting_tree(X,Y,D):
    '''
    创建单层提升树，找到误分类率最小的划分方式
    :param D: 前一轮训练数据权值分布
    :return: single_boosting_tree,单层提升树
    '''

    single_boosting_tree={}

    m,n=X.shape

    single_boosting_tree['e']=1 #初始化错误率 0<=e<=1
    for i in range(n): #遍历每一个特征，寻找最好的划分。
        for rule in ['LessIsOne','BiggerIsOne']: # 遍历每一种划分方式
            for div in [-0.5,0.5,1.5]: #遍历每一个划分点
                #计算e，和Gx
                tmp_Gx,tmp_e=cal_Gx_e(X,Y,div,rule,D,i)
                if tmp_e<single_boosting_tree['e']: #获得了更好的划分方式,保存
                    single_boosting_tree['e'] = tmp_e
                    single_boosting_tree['Gx'] = tmp_Gx
                    single_boosting_tree['div'] = div
                    single_boosting_tree['rule']=rule
                    single_boosting_tree['feature']=i

    single_boosting_tree['alpha']=1/2*np.log((1-single_boosting_tree['e'])/single_boosting_tree['e'])
    #返回单层提升树
    return single_boosting_tree

def create_boosting_tree(X,Y,tree_num=50):

    m,n=X.shape
    # 初始化权值，每个样本的权值是1/m
    D=np.array([1/m]*m) # 使用np.array便于计算

    Fx=[0]*m #用于计算当前分类器的输出 对应与公式8.6

    boosting_tree=[]
    for i in range(tree_num): #开始构造提升树
        single_boosting_tree=create_single_boosting_tree(X,Y,D)
        #根据上一次构造的单层提升树来更新误分类样本的权重
        # 需要熟悉np.array的运算
        # 举个🌰
        # a = np.array([1, 2, 3, 4])
        # b = np.array([1, 2, 3, 4])
        #
        # print(a * b)   输出[ 1  4  9 16]
        # print(np.sum(a * b))  输出 30

        # 计算规范化因子，对应于公式8.5
        Zm=np.sum(D*np.exp(-1*single_boosting_tree['alpha']*Y*single_boosting_tree['Gx']))
        # 计算下一轮D
        D=D/Zm*np.exp(-1*single_boosting_tree['alpha']*Y*single_boosting_tree['Gx'])

        boosting_tree.append(single_boosting_tree)


        # 当前线性预测值,对应公式8,6
        Fx+=single_boosting_tree['alpha']+single_boosting_tree['Gx']

        # 最终分类起预测值 公式8.7
        Gx=np.sign(Fx)
        # 总的错误个数
        total_error_num=np.sum([1 for i in range(m) if Gx[i]!=Y[i]])
        # 误差率
        total_error_rate=total_error_num/m

        #没有误差了，就可以直接返回
        if total_error_rate==0:
            return boosting_tree

        print(f'in {i}th epoch, error={single_boosting_tree["e"]}. total error is {total_error_rate}')

    return boosting_tree


def predict(x,tree):
    '''
    用于预测一个样本的输出
    :param x:
    :param tree: 提升树
    :return: GX，预测值
    '''
    fx = 0  # 分类器线性累加值

    for i in range(len(tree)):
        div=tree[i]['div']
        rule=tree[i]['rule']
        alpha=tree[i]['alpha']
        feature=tree[i]['feature']



        # 这里注意，每一个每类器最终预测的Gmx是+1，-1。
        # fx=sum（alpha*Gmx）
        # Gx=sign（fx）
        if rule=='LessIsOne':
            # 在LessIsOne规则下，小于div预测为1，大于预测为-1
            if x[feature]<div:
                fx+=alpha*1
            else:
                fx+=alpha*(-1)
        else: #BiggerIsOne
            if x[feature]<div:
                fx+=alpha*(-1)
            else:
                fx+=alpha*1

    Gx=np.sign(fx)
    return Gx

def test(X,Y,tree):
    acc = 0  # 正确率
    acc_num = 0  # 正确个数
    for i in range(len(X)):
        print('testing ***', i)
        Gx=predict(X[i],tree)
        if Gx == Y[i]:
            acc_num += 1
        print(f'testing {i}th data :y_pred={Gx},y={Y[i]}')
        print('now_acc=', acc_num / (i + 1))



if __name__=='__main__':

    # 获取当前时间
    start = time.time()

    # 读取训练文件
    print('load TrainData')
    X_train, y_train = loadData('../Mnist/mnist_train.csv')

    # 读取测试文件
    print('load TestData')
    X_test, y_test = loadData('../Mnist/mnist_test.csv')

    boosting_tree=create_boosting_tree(X_train[0:1000],y_train[0:1000],30)

    test(X_test[0:200],y_test[0:200],boosting_tree)

    end=time.time()

    print(end-start)



    # # 鸢尾花数据集 100%
    # from sklearn import datasets
    # from sklearn.model_selection import train_test_split
    #
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    #
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
    # y_train[y_train>0]=1
    # y_train[y_train==0]=-1
    # y_test[y_test > 0] = 1
    # y_test[y_test == 0] = -1
    #
    # boosting_tree=create_boosting_tree(X_train,y_train,10)
    # test(X_test,y_test,boosting_tree)







































