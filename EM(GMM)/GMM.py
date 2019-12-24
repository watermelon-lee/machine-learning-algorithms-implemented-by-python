"""
@File    : GMM.py
@Time    : 2019-12-24 14:45
@Author  : LEE
@Software: PyCharm
@Email: leehaoran@pku.edu.cn
"""


# 高斯混合模型
# EM算法一个简单但是重要的应用
# 关于高斯混合模型，可以用两个角度来看

# 几何角度
# 高斯混合模型中，每一个数据就是一个加权平均，是由多个高斯分布叠加而成的，即p=sum(alpha_k*Gaussian(y|theta_k) k=1,2...K

# 生成模型角度
# 数据产生过程如下
# 首先在多个模型中，依据概率alpha_k 选择一个模型 Gaussian_k
# 然后按照这个概率模型Gaussian(y|theta_k)随机产生一个观测值yi
# 反复上面的过程，就可以产生所有数据
# 所以混合模型数据中，每一个高斯模型的数据量 期望=总量*alpha



# 构造 高斯混合模型
# real_alpha_list = [0.1, 0.4, 0.5]
# real_mu_list = [3, -1, 0]
# real_sigma_list = [1, 4, 3]

# 预测值：
# alpha [0.07994016343715298, 0.4632026561031787, 0.45685718045966833]
# mu:[2.8466751203882925, -1.0023597711337926, 0.38938090877862847]
# sigma:[0.684965856763755, 3.869193761148044, 3.0488868940840943]


import numpy as np

def produce_data(alpha_list, mu_list, sigma_list, length):
    '''
    产生高斯混合模型数据
    :param alpha_list: 所有alpha的值
    :param m2: 所有mu的值
    :param sigma_list: 所有sigma的值
    :param length: 数据总长度
    :return: 返回高斯混合模型数据
    '''

    data=[] #所有数据



    for i in range(len(alpha_list)):
        # 设置一下随机种子，方便复现
        np.random.seed(3)
        data_i=np.random.normal(mu_list[i], sigma_list[i], int(length * alpha_list[i]))
        data.extend(data_i)

    return np.array(data)

def gaussian(y,mu,sigma):
    '''
    单个高斯模型的概率密度函数值
    :param y:观测数据
    :param mu:单个高斯模型的均值
    :param sigma:单个高斯模型的标准差
    :return: 单个高斯模型概率密度函数值
    '''

    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.square(y-mu)/(2*sigma**2))



def e_step(data,alpha_list, mu_list, sigma_list,length):
    '''
    计算gamma_jk
    :param data： 所有样本
    :param alpha_list: 所有alpha的值
    :param m2: 所有mu的值
    :param sigma_list: 所有sigma的值
    :param length:数据长度
    :return: gamma
    '''
    # 对应于算法9.2 E步
    # 计算响应度
    K=len(alpha_list) #模型个数
    # 使用ndarray便于计算
    # 注意是 0. 这样是float类型
    gamma=np.array([[0.]*K for _ in range(length)])

    for j in range(length):
        k=0
        for k in range(K):
            # 注意这里只算出了值，还没有作归一化
            # 需要算出所有的值之后才可以作归一化
            gamma[j][k]=alpha_list[k]*gaussian(data[j],mu_list[k],sigma_list[k])

        if k==K-1:
            # yj 对每个模型的响应度值都算出来了
            # 归一化
            # 使用切片一次更新
            gamma[j,:]=gamma[j,:]/sum(gamma[j])
    # 返回响应度
    return gamma

def m_step(data,gamma,alpha_list,mu_list,sigma_list):
    '''
    # 对应于算法9.2M步
    :param data:
    :param gamma: 响应度
    :param mu_list:
    :param sigma_list:
    :param alpha_list:
    :return: 更新之后的参数
    '''

    # 纯照打公式
    for k in range(len(alpha_list)):

        # 由于跟新sigma需要用到久的 mu，所有在更新mu 之前更新
        sigma_list[k]=np.sqrt(np.sum(gamma[:,k]@np.square(data-mu_list[k]))/sum(gamma[:,k]))

        mu_list[k]=np.sum(gamma[:,k]@data)/np.sum(gamma[:,k])

        alpha_list[k]=(np.sum(gamma[:,k])/len(data))

    return alpha_list,mu_list,sigma_list

def EM_for_GMM(data,epoch=500):
    # 算法第一步，取初始值
    # 可以随机选取，但注意，EM算法最后并不一定得到全局最优解
    # 初始值的选取可能会对算法结果有较大的影响
    alpha_list=[0.2,0.4,0.4]
    mu_list=[0,-2,2]
    sigma_list=[1,2,3]

    for i in range(epoch):
        # 反复迭代2，3步骤


        gamma=e_step(data,alpha_list,mu_list,sigma_list,len(data))
        alpha_list,mu_list,sigma_list=m_step(data,gamma,alpha_list,mu_list,sigma_list)
        if i%100==0:
            print(f'epoch={i}')
            print(f'alpha={alpha_list}')
            print(f'mu={mu_list}')
            print(f'sigma={sigma_list}')

    # 返回参数
    return alpha_list,mu_list,sigma_list

if __name__=='__main__':

    # 设置缓和高斯模型参数，以生成数据

    real_alpha_list = [0.1, 0.4, 0.5]
    real_mu_list = [3, -1, 0]
    real_sigma_list = [1, 4, 3]

    print(f'real model parameter is: alpha {real_alpha_list};mu:{real_mu_list};sigma:{real_sigma_list}')
    data=produce_data(real_alpha_list,real_mu_list,real_sigma_list,2000)

    alpha,mu,sigma=EM_for_GMM(data)

    print(f'predict model parameter is: alpha {alpha};mu:{mu};sigma:{sigma}')
































