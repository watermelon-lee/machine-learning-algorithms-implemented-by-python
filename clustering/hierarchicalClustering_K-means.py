"""
@File    : hierarchicalClustering_K-means.py
@Time    : 2019-12-07 12:55
@Author  : 李浩然
@Software: PyCharm
@Email: leehaoran@pku.edu.cn
"""


# 测试数据 iris.txt 鸢尾花数据集 150 条
# 输出：{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 1, 51: 1, 52: 2, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 1, 67: 1, 68: 1, 69: 1, 70: 1, 71: 1, 72: 1, 73: 1, 74: 1, 75: 1, 76: 1, 77: 2, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 1, 84: 1, 85: 1, 86: 1, 87: 1, 88: 1, 89: 1, 90: 1, 91: 1, 92: 1, 93: 1, 94: 1, 95: 1, 96: 1, 97: 1, 98: 1, 99: 1, 100: 2, 101: 1, 102: 2, 103: 2, 104: 2, 105: 2, 106: 1, 107: 2, 108: 2, 109: 2, 110: 2, 111: 2, 112: 2, 113: 1, 114: 1, 115: 2, 116: 2, 117: 2, 118: 2, 119: 1, 120: 2, 121: 1, 122: 2, 123: 1, 124: 2, 125: 2, 126: 1, 127: 1, 128: 2, 129: 2, 130: 2, 131: 2, 132: 2, 133: 1, 134: 2, 135: 2, 136: 2, 137: 2, 138: 1, 139: 2, 140: 2, 141: 2, 142: 1, 143: 2, 144: 2, 145: 2, 146: 1, 147: 2, 148: 2, 149: 1}
# Counter({1: 62, 0: 50, 2: 38})
# 实际是0-49被准确归为0了
# 正确的分类应该是 （0-49） （50-99） （100-149） 每种50





# 因为K-means聚类结果很大程度上会受到初始值的影响
# 所以先使用层次聚类分好K个类
# 然后从每个类中选取一个与中心距离最近的点
# 将这些点作为k-means的初始中心点，在进行聚类


import numpy as np
import collections

# 计算两个样本点之间的距离
def calDist(x1,x2):
    # 使用欧式距离
    return np.sum(np.square(x1-x2))

def cal_Cluster_distance(D,cluster1,cluster2):
    '''
    计算两个蔟之间的距离，使用的类间距离是最短距离
    :param D: 样本距离矩阵
    :param cluter1:
    :param cluster2:
    :return: 两个蔟之间最短距离
    '''
    minDist=10000
    for i in cluster1:
        for j in cluster2:
            if D[i][j]<minDist:
                minDist=D[i][j]
    return minDist


def find_miniDist_index(D,cluster):
    '''
    计算最短距离的两个类
    :param D: 样本距离矩阵
    :param cluster: 所有类
    :return: 需要合并两个类的索引
    '''
    # 开始遍历每一个类，找到类间距离最小的两个类，进行合并
    # 类间距离极为两个类中间最近的两个点的距离，即最短距离
    minDist=10000
    indexi,indexj=-1,-1
    for i in cluster.keys():
        for j in cluster.keys():
            #计算两个蔟之间的距离
            if i!=j:
                distance=cal_Cluster_distance(D,cluster[i],cluster[j])
                #保存最短距离的索引
                if distance<minDist:
                    minDist=distance
                    indexi=i
                    indexj=j

    return indexi,indexj


def hierarchical_clustering(data,k):
    '''
    层次聚类首先将每一个样本作为一类
    然后选取距离最短的两个样本作为一类
    然后重新进行聚类，知道样本被分为制定类数

    使用欧式距离作为样本之间距离，使用最短距离为类间距离

    :param data: 样本点 np.array
    :param k: 需要聚类的蔟数
    :return: 聚类之后每个离每个蔟中心最近的k个初始点
    '''

    # 我们使用一个字典保存属于每一类的有哪些样本点
    # 首先将样本每一个点分为一类
    cluster={}
    for i in range(len(data)):
        cluster[i]=[i]

    # 样本距离矩阵
    D=[[0 for _ in range(len(data))] for _ in range(len(data))]

    # 计算样本距离矩阵
    # 可知距离矩阵为对称矩阵，只需计算上三角部分
    for i in range(len(data)):
        for j in range(i,len(data)):
            distance=calDist(data[i],data[j])
            D[i][j]=distance
            D[j][i]=distance

    clusters=len(cluster) # cluster的长度即为其中的类别个数

    while clusters>k:
        print(f'clustering *** the cluster num is {clusters}')
        i,j=find_miniDist_index(D,cluster)
        # 将i，j两个蔟合并到i，删除j
        # 使用extend，将cluster[j]中的元素逐个添加到cluster[i]中
        cluster[i].extend(cluster[j])
        # 删除j类
        del cluster[j]
        clusters=len(cluster)

    print(f'hierarchical clustering：{cluster}')
    # 样本分成了k类，返回k个最接近每个蔟中心点的样本作为初始中心点
    initial_start=[]
    for i in cluster.keys():
        # 计算这一类样本的中心点
        center=np.array([0. for _ in range(data.shape[1])])
        for j in range(len(cluster[i])):
            # 先计算所有样本点的总和，在除以个数
            center+=data[cluster[i][j]]
        center/=len(cluster[i])

        # 找到离中心点最近的点
        miniDist=10000
        index=-1
        for j in range(len(cluster[i])):
            tmp=calDist(center,data[cluster[i][j]])
            if tmp<=miniDist:
                index=cluster[i][j]
                miniDist=tmp
        initial_start.append(index)

    # 返回样本点的索引
    print(f'hierarchical clustering find start：{initial_start}')
    return initial_start


def loadData(filename):

    data=[]
    with open(filename) as file:
        # 按行读取
        for line in file.readlines():
            # 按照逗号切分
            # 对于最后一个元素是鸢尾花名字，我们不需要
            line=line.split(',')[:-1]
            linedata=[]

            for i in range(len(line)):
                linedata.append(eval(line[i]))
            linedata=np.array(linedata)
            data.append(linedata)
    # 返回ndarray，便于以后计算
    return np.array(data)


def k_means(start, data, k):
    '''
    对于当前类的中心，每个样本计算到所有蔟中心点的距离，然后划分到最近的蔟
    划分结束之后更新每一类，得到新的类中心点。持续计算，直到收敛（每一个样本分类不再变化）
    :param start: k个起始点序号
    :param data: 样本点
    :param k: 分类个数
    :return: 分类之后的序号
    '''
    m,n=data.shape # 样本数量与特征数
    cluster={}#用于保存分类样本点
    cluster_center={}# 保存分类的中心点

    #初始化,以初始点为中心开始聚类
    for i in range(m):
        cluster[i]=-1 #保存每一个样本点所属分类，初始没有分类，设为-1
    for i in range(k):
        cluster_center[i]=data[start[i]]

    # 样本类别改变的数量
    # 当每一个样本类别数量不再改变之后，就认为聚类结束
    changed_data=1

    while changed_data:
        # 将其置为0，之后改变了就继续循环
        changed_data=0
        # 遍历每一个样本点，计算其离哪一个蔟中心点更近，将其分到那一类
        for i in range(m):
            minDist=10000
            cluster_belong=-1 # 所属的类别
            # 计算与每一个中心点的距离
            for c in range(len(cluster_center)):
                distance=calDist(cluster_center[c],data[i])
                if distance<minDist:
                    minDist=distance
                    cluster_belong=c
            # 遍历完成，进行比较，是否改变类别
            if cluster_belong!=cluster[i]:
                changed_data+=1 #有一个样本点类别改变
                cluster[i]=cluster_belong


        # 计数器，计算每个类别有多少样本
        count=[0 for _ in range(k)]
        # k的样本中心，初始设为0
        # 使用ndarray便于使用广播1机制进行计算
        center=[np.array([0. for _ in range(n)]) for _ in range(k)]
        for index, c in cluster.items():
            # index 为0-m，表示样本的序号
            # c 0，1，2。表示的样本的类别
            # 计算得到每一类样本点的和，还需要需要除以个数得到中心点
            center[c]+=data[index]
            count[c]+=1

        # 修改中心
        for i in range(k):
            cluster_center[i]=center[i]/count[i]

    return cluster







if __name__=='__main__':
    data=loadData('iris.txt')
    start=hierarchical_clustering(data,3)


    # 之间随机选取中心点
    # 使用随机选取几乎无法超过先层次聚类选取中心点
    # 可能是样本中有些值确实很难正确区分
    # 毕竟无监督算法聚类难度是大于有监督的预测

    # start1=[np.random.randint(0,150),np.random.randint(0,150),np.random.randint(0,150)]
    cluster=k_means(start,data,3)



    print(cluster)

    print(collections.Counter(cluster.values()))

























