"""
@File    : PCA.py
@Time    : 2019-12-10 09:54
@Author  : 李浩然
@Software: PyCharm
@Email: leehaoran@pku.edu.cn
"""


# PCA的实现比较简单
# 复杂的点在于明白其中的原理与推导吧

# PCA的实现有两种方式，以下是基于SVD奇异值分解法

import numpy as np

def normalization(X):
    '''
    对样本矩阵进行标准化 式16.48
    :param X: 样本矩阵（mxn）m为纬度，n为样本数
    :return: 标准化之后的样本矩阵
    '''
    m,n=X.shape
    X_mean=np.mean(X,axis=1) #求每一行对平均值
    S=[0 for _ in range(m)]
    for i in range(m):
        S[i]=(1/(n-1)*np.sum(np.square(X[i]-X_mean[i])))
        for j in range(n):
            X[i][j]=(np.float(X[i][j])-X_mean[i])/S[i]
    return X

def nor(X):
    X_mean=np.mean(X,axis=1) #求每一行对平均值
    for i in range(len(X)):
        X[i]=X[i]-X_mean[i]
    return X


def createX(X):
    '''
    对应《统计学方法》p316算法第一步
    :param X: 标准化之后样本矩阵 mxn
    :return: nxm矩阵X'
    '''
    _,n=X.shape
    return 1/(np.sqrt(n-1))*X.T

def svd(X,target=0.85):
    '''
    进行svd奇异值分解，得到满足保留样本特征target百分比前k个特征向量
    :param target: 希望保留成分百分比，默认百分之85
    :return: VT的前k列构成的矩阵
    '''
    U,sigma,VT=np.linalg.svd(X)
    # U:nxn sigma:n*m,VT:m*m
    lambdas=[sigma[i] for i in range(len(sigma))]
    k=0
    # 遍历奇异值，找到加起来满足target的前k个值
    total=sum(lambdas)
    for i in range(1,len(lambdas)):
        if sum(lambdas[:i])/total>target:
            k=i
            break
    # 返回V的前k列构成的矩阵
    # VT.T 就是V
    return VT.T[:,:k]

def svd_k(X,k):
    '''
    :param X: 样本矩阵
    :param k: k个主成分
    :return:  返回前k个特征向量
    '''
    U,sigma,VT=np.linalg.svd(X)
    return VT.T[:,:k]


if __name__=='__main__':
    from sklearn.decomposition import PCA

    X = np.array([[-1., 1.,3.], [-2., -1.,4.], [-3., -2.,5.], [1., 1.,-2.], [2., 1.,-4.], [3., 2.,-5.]])
    pca = PCA(n_components=2)
    pca.fit(X)
    pca.transform(X)
    print(pca.transform(X))


    # 因为实现算法的时候，X样本用的是mxn m代表纬度，n代表样本数。
    # 与sklearn中维度相反，所以计算的时候使用X.T
    # sklearnfit中，X为n*m n代表样本数，m代表纬度
    X_norm=nor(X.T)
    X_new=createX(X_norm)
    VT=svd_k(X_new,2)
    VT2=svd(X_new,0.85)

    print(VT.T@X.T)
    print(VT2.T@X.T)



    # 可以发现，标准化不同，得到的最后主成分也有不同
    X_norm = normalization(X.T)
    X_new = createX(X_norm)
    VT = svd_k(X_new, 2)
    VT2 = svd(X_new, 0.85)

    print(VT.T @ X.T)
    print(VT2.T @ X.T)





