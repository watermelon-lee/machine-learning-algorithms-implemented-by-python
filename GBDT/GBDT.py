"""
@File    : GBDT.py
@Time    : 2019-12-20 14:09
@Author  : Lee
@Software: PyCharm
@Email: leehaoran@pku.edu.cn
"""

import pandas as pd
import numpy as np


# 样本 boston_house_price

# 对数据集的说明：
# CRIM：城镇人均犯罪率。
# ZN：住宅用地超过 25000 sq.ft. 的比例。
# INDUS：城镇非零售商用土地的比例。
# CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
# NOX：一氧化氮浓度。
# RM：住宅平均房间数。
# AGE：1940 年之前建成的自用房屋比例。
# DIS：到波士顿五个中心区域的加权距离。
# RAD：辐射性公路的接近指数。
# TAX：每 10000 美元的全值财产税率。
# PTRATIO：城镇师生比例。
# B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
# LSTAT：人口中地位低下者的比例。
# MEDV：自住房的平均房价，以千美元计。


# 测试结果
# 以百分之20误差率以内作为一次良好的预测
# 测试准确率80%左右
# R2_score（相关系数的平方）在0.7左右（不是很稳定，因为样本数量较小，测试数据在100条左右）
# 调整迭代轮数（树的数量）以及树的深度（max_deep)，以及对残差拟合回归树的正则化系数（alpha）都会有不同的影响


def loaddata(filename):
    data=pd.read_csv(filename)

    #打乱数据集
    from sklearn.utils import shuffle
    data = shuffle(data)
    # 你也可以设置种子保证结果的一致
    # data = shuffle(data,random_state=666)

    # 切分样本，作为训练集和测试集
    rate=0.2
    data=data.values
    train_data=data[:int(len(data)*(1-rate)),:]
    test_data=data[int(len(data)*(1-rate)):,:]
    return train_data,test_data


def findBestFeatureAndPoint(node):
    '''
    依据MSE准则，找到最佳切分特征和最佳切分点
    :param node: 进行分裂的节点, 一个矩阵
    :return: 切分特征与切分点
    '''

    # n为特征数
    m,n=node.shape
    # 因为最后一列是标签值
    n=n-1
    # 需要预测的真实值
    y = node[:, -1]

    # 用来保存最佳切分特征与切分点
    # 以及左右子树
    min_loss = np.Inf
    best_feature = -1
    best_point = -1
    best_left=None
    best_right=None



    # 找到最佳切分特征与切分点
    # 我们遍历所有特征，然后遍历该特征所有（或者部分）切分点
    # 取决于该特征是离散还是连续变量
    for feature in range(n):
        # 注意是n-1 ， 因为最后一个是样本需要预测的值

        # 获得进行切分列
        # 因为是连续数据，有可能有很多不同的值
        # 所以此处我们进行切分的时候，若是离散数据（默认种类小于等于10），我们进行精确切分
        # 若类型大于10，认为是连续变量，进行10分位点切分
        column=node[:,feature]
        category=sorted(set(column))
        if len(category)<=10:
            split_point=category
        else:
            # 使用np.arrange来每次找到1/10数据点所在的索引
            # 然后进行切分
            split_point = np.arange(0, len(category), len(category) // 10)
            split_point = [category[split_point[i]] for i in range(0, len(split_point))]



        # 确定了所有切分点之后，对切分点进行遍历，找到最佳切分点
        for point in split_point:
            # 尝试切分
            left=column[column<=point]
            right=column[column>point]

            # 左右两边的需要预测的真实值
            y_left=y[column<=point]
            y_right=y[column>point]
            # 计算左右两边最佳的Cmj
            # cart回归树损失函数为MSE
            # 所以我们只需要取节点上的均值即可

            c_left = np.average(y_left)
            c_right = np.average(y_right)

            loss=np.sum(np.square(y_left-c_left))+np.sum(np.square(y_right-c_right))
            if loss<min_loss:
                min_loss=loss
                best_feature=feature
                best_point=point
                best_left=node[column<=point]
                best_right=node[column>point]
    return (best_feature,best_point,best_left,best_right)







def createCART(data,deep,max_deep=2):
    '''
    创建回归树，分裂准则MSE（最小均方误差）
    :param deep: 树的当前深度
    :param max_deep:  树的最大深度（从0开始），默认为2，即产生4个叶子节点
    :param data: 训练样本，其中data中的最后一列值为上一轮训练之后的残差
    :return: 一颗回归树
    '''

    # 树的结构例如
    # tree={3:{'left':{4:{'left':23.1,'right':19.6},'point':0},'right':{6:{'left':23.1,'right':19.6},'point':4.5}},'point':10.4}
    # 上面是一颗2层的回归树
    # 3代表根节点以第三个特征进行分类，分裂的切分点是point=10.4
    # 然后是左右子树left，right
    # left也是一个字典，对应左子树
    # 4代表左子树以特征四为分裂特征，切分点是point=0
    # 分裂之后的left仍然是一个字典，其中有left和right对应着23.1,19.6
    # 这两个值即为我们的预测值
    # 右子树也同理

    if deep<=max_deep:
        feature,point,left,right=findBestFeatureAndPoint(data)
        tree = {feature: {}}
        if deep!=max_deep:
            # 不是最后一层，继续生成树
            tree['point']=point
            if len(left)>=2:
                # 必须要保证样本长度大于1，才能分裂
                tree[feature]['left']=createCART(left,deep+1,max_deep)
            else:
                tree[feature]['left']=np.average(left)
            if len(right)>=2:
                tree[feature]['right']=createCART(right,deep+1,max_deep)
            else:
                tree[feature]['right']=np.average(right)

        else:
            # feature, point, left, right = findBestFeatureAndPoint(data)
            # tree['point']=point
            # # y标签在训练样本最后一列，用-1获取
            # y_left=left[:,-1]
            # y_right=right[:,-1]
            # c_left = np.average(y_left)
            # c_right = np.average(y_right)

            # 最后一层树，保存叶节点的值
            return np.average(data[:,-1])
        return tree


def gradientBoosting(round, data, alpha):
    '''

    :param round: 迭代论数，也就是树的个数
    :param data: 训练集
    :param alpha: 防止过拟合，每一棵树的正则化系数
    :return:
    '''

    tree_list=[]
    # 第一步，初始化fx0，即找到使得损失函数最小的c
    # 即所有样本点的均值
    # -1 代表没有切分特征，所有值均预测为样本点均值
    fx0={-1:np.average(data[:,-1])}

    tree_list.append(fx0)
    # 开始迭代训练，对每一轮的残差拟合回归树
    for i in range(1,round):
        # 更新样本值，rmi=yi-fmx
        # TODO:没有想到更新残差较好的方式
        #  目前想到的就是对每一个样本以当前的提升树进行一次预测
        #  然后获得预测值与真实值进行相减，将样本真实值变为残差
        #  如果你碰巧看到了，有好的想法，欢迎与我交流～
        if i==1:
            data[:,-1]=data[:,-1]-fx0[-1]
        else:
            for i in range(len(data)):
                # 注意，这里穿的列表是tree_list中最后一个
                # 因为我们只需要对残差进行拟合，data[:,-1]每一轮都进行了更新，所以我们只要减去上一颗提升树的预测结果就是残差了
                data[i, -1] = data[i, -1] - predict_for_rm(data[i], tree_list[-1], alpha)
        # 上面已经将样本值变为了残差，下面对残差拟合一颗回归树
        fx = createCART(data, deep=0, max_deep=4)
        #
        # 将树添加到列表
        tree_list.append(fx)
    return tree_list


def predict_for_rm(data, tree, alpha):
    '''
    获得前一轮 第m-1颗树 的预测值，从而获得残差
    :param data: 一条样本
    :param tree: 第 m-1 颗树
    :param alpha: 正则化系数
    :return:  第m-1颗树预测的值
    '''

    while True:
        # 遍历该棵树，直到叶节点
        # 叶节点与子树的区别在于一节点上的值为float
        # 而子树是一个字典，有point键，用作切分点
        # tree={3:{'left':{4:{'left':23.1,'right':19.6},'point':0},'right':{6:{'left':23.1,'right':19.6},'point':4.5}},'point':10.4}
        #
        if type(tree).__name__=='dict':
            # 如果是字典，那么这是一颗子树,
            point = tree['point']
            # tree.keys()=dict_keys([3, 'point'])
            # 所以int值对应的是特征，但是字典的键值是无序的，我们无法保证第一个是特征，所以用类型来判断
            feature = list(tree.keys())[0] if type(list(tree.keys())[0]).__name__ == 'int' else list(tree.keys())[1]
            if data[feature] <= point:
                tree = tree[feature]['left']
            else:
                tree = tree[feature]['right']
        else:
            # 当tree中没有切分点point，证明这是一个叶节点，tree就是预测值，返回获得预测值
            return alpha * tree




def predict(data, tree_list, alpha):
    '''
    对一条样本进行预测
    :param tree_list: 所有树的列表
    :param data: 一条需要预测的样本点
    :param alpha:正则化系数
    :return: 预测值
    '''
    m=len(tree_list)
    fmx=0
    for i in range(m):
        tree=tree_list[i]
        if i==0:
            #  fx0={-1:np.average(data[:,-1])}
            # fx0是一个叶节点，只有一个预测值，树的深度为0
            fmx+=tree[-1]
        else:
            while True:
                # 遍历该棵树，直到叶节点
                # 叶节点与子树的区别在于一节点上的值为float
                # 而子树是一个字典，有point键，用作切分点
                # tree={3:{'left':{4:{'left':23.1,'right':19.6},'point':0},'right':{6:{'left':23.1,'right':19.6},'point':4.5}},'point':10.4}
                #
                if type(tree).__name__=='dict':
                    # 如果是字典，那么这是一颗子树,
                    point=tree['point']
                    # tree.keys()=dict_keys([3, 'point'])
                    # 所以int值对应的是特征，但是字典的键值是无序的，我们无法保证第一个是特征，所以用类型来判断
                    feature=list(tree.keys())[0] if type(list(tree.keys())[0]).__name__=='int' else list(tree.keys())[1]
                    if data[feature]<=point:
                        tree=tree[feature]['left']
                    else:
                        tree=tree[feature]['right']
                else:
                    # 当tree中没有切分点point，证明这是一个叶节点，tree就是预测值，返回获得预测值
                    fmx+= alpha * tree
                    break
    return fmx


def test(X_test, y_test, tree_list, alpha):
    acc = 0  # 正确率
    acc_num = 0  # 正确个数
    y_predict=[]
    for i in range(len(X_test)):
        print('testing ***', i)
        x = X_test[i]
        y_pred =predict(x, tree_list, alpha)
        y_predict.append(y_pred)
        if y_pred/y_test[i]<1.25 and y_pred/y_test[i]>0.8:
            acc_num += 1
        print(f'testing {i}th data :y_pred={y_pred},y={y_test[i]}')
        print('now_acc=', acc_num / (i + 1))
    return y_predict


if __name__=='__main__':
    train_data,test_data=loaddata('boston_house_prices.csv')


    tree_list=gradientBoosting(10,train_data,0.12)

    X_test,y_test=test_data[:,:-1],test_data[:,-1]

    y_pred=test(X_test,y_test,tree_list,0.12)

    from sklearn.metrics import r2_score

    score = r2_score(y_test, y_pred)
    print(score)




