"""
@File    : SVM.py
@Time    : 2019-12-02 19:57
@Author  : 李浩然
@Software: PyCharm
@Email: leehaoran@pku.edu.cn
"""

# mnist_train:60000
# mnist_test:500

# 实际使用训练集2000
# 测试集400
# time:115.8s
# acc:0.93



import pandas as pd
import numpy as np
import time



def loadData(fileName):

    data=pd.read_csv(fileName,header=None)


    #将dataframe转化为numpy.array
    data=data.values

    #从样本中切分出分类结果
    y_label = data[:, 0]
    X_label = data[:, 1:]/255 #转化为0-1之间的数

    # 使用SVM进行二分类
    # 将数据分为两类，大于0为1，等于0为-1
    y_label[y_label>0]=1
    y_label[y_label==0]=-1

    # 按照5分界，样本点很难找出一个较好的超平面区分开，正确率只有80多
    # y_label[y_label<5]=0
    # y_label[y_label>=5]=1

    return X_label,y_label


class SVM:
    def __init__(self,X,Y,sigma=10,C=200,toler=0.0001):
        '''

        :param X: 训练样本
        :param Y: 样本标签
        :param sigma: 高斯核函数的sigma（标准差）
        :param C: 对每一个松弛变量的惩罚系数
        :param toler: 松弛变量
        '''

        self.m,self.n=X.shape  #m为样本数（60000），n为特征数（784=28*28）
        self.X=X
        self.Y=Y#转化为列向量
        self.sigma=sigma
        self.C=C
        self.toler=toler
        self.E=[-Y[i] for i in range(len(Y))] #误差，为g(x)-yi。初始gx=0，设置为-yi
        self.alpha=[0 for _ in range(len(Y))] #每一个样本对应的拉格朗日系数alpha
        self.K=self.calKernel() # 高斯核矩阵，存放代替内积的核函数值
        self.b=0 #偏置
        self.supportVector=[]#支持向量的索引，用于预测。因为W只和alpha>0的样本有关。

    def calKernel(self):
        '''
        #使用高斯核函数（式7.90）代替内积xi*xj
        :return:高斯核矩阵
        '''
        print('construct the Gaussion kernel matrix')
        K=[[0 for _ in range(self.m)] for _ in range(self.m)]
        for i in range(self.m): #遍历每一个样本
            if i%100==0:
                string="*"*(i//100+1)
                print('constructing'+string)
            xi=self.X[i]
            for j in range(i,self.m):
                #可以知道高斯核矩阵是对称矩阵，K【i】【j】=k【j】【i】
                #所以我们只需要计算上三角矩阵的值
                xj=self.X[j]
                Kij=np.exp(-1*(xi-xj)@(xi-xj).T/(2*self.sigma**2)) #xi-xj为1*784维的行向量
                K[i][j],K[j][i]=Kij,Kij
        return K

    def calGx(self,i):
        '''
        用于计算Gx，式7.104
        :param i: 第i个样本
        :return: Gxi 表示SVM的Gx函数对xi的预测值
        '''
        Gxi=0

        #对应公式7.104。计算出预测值。其实也就是Wx+b
        # for j in range(self.m):
        #     Gxi+= self.alpha[j] * self.Y[j] * self.K[i][j] + self.b
        # 优化，我们只需要计算alpha不为0的即可
        index=[i for i in range(self.m) if self.alpha[i]!=0]
        for j in index:
            Gxi+= self.alpha[j] * self.Y[j] * self.K[j][i]
        Gxi+=self.b # 别忘了加上偏置
        return Gxi

    def calE(self,i):
        '''
        计算出误差，对应于公式7.105 Ei=Gxi-yi
        :param i: 第i个样本
        :return:
        '''
        Gxi=self.calGx(i) #计算Gxi
        Ei=Gxi-self.Y[i]
        return Ei

    def alpha1_break_KTT(self,i):
        '''
        注意，该检验是在误差为toler范围内检验的
        优先选择满足条件0<alpha<C的样本点（支持向量）
        并没有做比较谁更严重，只找到第一个就返回
        :param: 第i个样本
        :return: bool,是否违反KTT条件。true为违反了
        '''

        yi=self.Y[i]
        Gxi=self.calGx(i)
        alpha1=self.alpha[i]

        if self.alpha[i]>-self.toler and self.alpha[i]<self.C+self.toler \
                and np.abs(yi*Gxi-1)<=self.toler:
            return False
        elif np.abs(alpha1-self.C)<=self.toler and yi*Gxi<=1:
            return False
        elif np.abs(alpha1)<=self.toler and yi*Gxi>=1:
            return False
        return True #违反了KTT条件

    def getAlpha2(self,i,Ei):
        # 我们要找到误差最大的两个alpha，即｜E1-E2｜最大
        # 那么当E1大于0，我们就需要找到最小的E2，
        # 如果E1是负的，我们就需要找到最大的E2
        '''
        :param i: 第i个样本点
        :param Ei:
        :return: ｜E1-E2｜最大的index
        '''

        if Ei>=0:
            index=self.E.index(min(self.E))
            return index,self.calE(index) #返回最小的E2，和其index，
        else:
            index = self.E.index(max(self.E))
            return index, self.calE(index)  # 返回最大的E2，和其index，

    def train(self,epoch=100):


        step = 0 #迭代步数

        alphaChanged = 1
        # 改变的alpha数量。用于判断是否提前结束
        # 当这一轮循环没有alpha改变，那么我们认为得到了最优的alpha，结束训练/

        while step<epoch and alphaChanged>0:

            print(f'in the {step}th step, training****')
            step+=1
            alphaChanged=0#将其修改为0，如果没有变化为，就可以结束训练

            # SMO算法，选择第一个alpha变量
            # 选取第一个变量为外层循环，选取破坏第一个违反KTT条件最为严重的样本点。
            for i in range(self.m):
                if self.alpha1_break_KTT(i):  # 如果违反了KTT条件

                    E1=self.calE(i) #需要重新计算Ei，应为之前的alpha变化了的话，Ei也会变。

                    j,E2 = self.getAlpha2(i, E1)  #找到｜E1-E2｜最大的alpha2
                    alpha1 = self.alpha[i]
                    alpha2 = self.alpha[j]

                    if self.Y[i] == self.Y[j]:
                        k = alpha1 + alpha2
                        L = max(0, k-self.C)
                        H = min(self.C, k)
                    else:
                        k = alpha1 - alpha2
                        L = max(0, -k)
                        H = min(self.C, self.C-k)

                    if L==H: #当L=H的时候，参数不会更新了，直接下一次迭代。
                        continue
                    # 更新alpha2
                    # alpha2-new=alpha2+(y2(E1-E2))/Eta 公式7.106
                    # Eta=K11+K22-2K12 公式7.107

                    eta = self.K[i][i] + self.K[j][j] - 2 * self.K[i][j]
                    alpha2_new = alpha2 + self.Y[j] * (E1-E2) / eta

                    if alpha2_new < L:
                        alpha2_new = L
                    elif alpha2_new > H:
                        alpha2_new = H

                    # 根据式7.109跟新alpha1
                    alpha1_new = alpha1 + self.Y[i] * self.Y[j] * (alpha2 - alpha2_new)

                    # 更新b 式子7.116,7.115,
                    b1_new=-1*E1-self.Y[i]*self.K[i][i]*(alpha1_new-alpha1)-self.Y[j]*self.K[j][i]*(alpha2_new-alpha2)+self.b
                    b2_new=-1*E2-self.Y[i]*self.K[i][j]*(alpha1_new-alpha1)-self.Y[j]*self.K[j][j]*(alpha2_new-alpha2)+self.b

                    if alpha1_new>0 and alpha1_new<self.C:
                        self.b=b1_new
                    elif alpha2_new>0 and alpha2_new<self.C:
                        self.b=b2_new
                    else:
                        self.b=(b1_new+b2_new)/2

                    #更新状态
                    self.alpha[i]=alpha1_new
                    self.alpha[j]=alpha2_new

                    self.E[i]=self.calE(i)
                    self.E[j]=self.calE(j)


                    # 如果α2的改变量过于小，就认为该参数未改变，alphaChanged不增加
                    # 反之则自增1
                    if np.abs(alpha2_new-alpha2)>=1e-5:
                        alphaChanged+=1

                    print(f'step num:{step},changed alpha num: {alphaChanged}')
         # 遍历所有alpha，将alpha>0的支持向量索引放入self.supportVector中
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.supportVector.append(i)




    def guassianKernel(self,xi,xj):
        '''
        用于预测的时候计算两个样本点的高斯核函数值
        :param i: 样本Xi
        :param j: 样本Xj
        :return: Kij
        '''
        return np.exp(-1*(xi-xj)@(xi-xj).T/(2*self.sigma)) #xi-xj为1*784维的行向量

    def predict(self,x):
        Gx=0
        # for i in range(self.m):
        #     Gx+= self.alpha[i] * self.Y[i] * self.guassianKernel(x,self.X[i])

        # 只需要遍历alpha不为0的支持向量即可
        for i in self.supportVector:
            Gx += self.alpha[i] * self.Y[i] * self.guassianKernel(x, self.X[i])

        return np.sign(Gx+self.b)

    def test(self,X_test, y_test):
        acc = 0  # 正确率
        acc_num = 0  # 正确个数
        for i in range(len(X_test)):
            print('testing ***',i)
            x = np.mat(X_test[i])
            y_pred = self.predict(x)
            if y_pred == y_test[i]:
                acc_num += 1
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

    print('Init SVM classifier')
    svm=SVM(X_train[0:2000],y_train[0:2000])

    print('start to train')
    svm.train(100)

    print('start to test')
    svm.test(X_test[0:400], y_test[0:400])

    # 获取结束时间
    end = time.time()

    print('run time:', end - start)


























































































