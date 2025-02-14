import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_SIZE = 500
buckets = 50
# 第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
plt.subplot(1, 2, 1)
plt.xlabel("random.lognormalvariate")
mu = -0.6
sigma = 0.15  # 将输出数据限制到0-1之间
diff1=[]
for i in range(10):
    diff1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])
plt.hist(diff1[0], buckets)  # 直方图 第一个参数是数据，第二个是直方图的组数 记住这些画图的参数 要积累了


# 第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
plt.subplot(1, 2, 2)
plt.xlabel("random.betavariate")
alpha = 1
beta = 10
diff2=[] 
for i in range(10):
    diff2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])
plt.hist(diff2[0], buckets)
plt.show()
'''
# x = np.random.randn(500)
x = [1, 2]
y = [2, 2]
plt.scatter(x, y, c='r')
plt.show()'''

# Numpy版本
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
      sum(kernel_val): 多个核矩阵之和
    '''
    # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = np.concatenate((source, target), axis=0)
    # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
    total0 = np.expand_dims(total, axis=0)
    total0 = np.broadcast_to(total0, [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按列复制
    total1 = np.expand_dims(total, axis=1)
    total1 = np.broadcast_to(total1, [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = np.sum(np.square(total0-total1), axis=2)  # ##ERROR：上版本为np.cumsum，会导致计算出现几何倍数的L2
    # 对角线的是样本自身的距离，其他则是不同元素之间的距离
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    print(bandwidth_list)
    # 高斯核函数的数学表达式
    kernel_val = [np.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # 多核合并


def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
     loss: MK-MMD loss
    '''
    batch_size = int(source.shape[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    print(kernels)
    # 将核矩阵分成4部分
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    # 这里计算出的n_loss是每个维度上的MK-MMD距离，一般还会做均值化处理
    n_loss= loss / float(batch_size)
    return np.mean(n_loss)


# ###----------------------------
# 测试上述函数
# ###----------------------------
SAMPLE_SIZE = 500
buckets = 50

# 第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
mu = -0.6
sigma = 0.15  # 将输出数据限制到0-1之间
diff1=[]
for i in range(10):
    diff1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])
# 第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
alpha = 1
beta = 10
diff2=[]
for i in range(10):
    diff2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])

MK_MMD(np.array(diff1),np.array(diff2))  # #经调整 该计算结果为6.047

# pytorch版本
import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)  # 差值平方然后在第二个维度求和就是每一行加一块，也就是某个样本与自己和其他样本的距离之和
    # 然后优化距离最小即可
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    # kernels是所哟有的距离计算，然后可以分成四块 XX XY YX YY
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

# ###----------------------------
# 测试上述函数
# ###----------------------------


SAMPLE_SIZE = 500
buckets = 50

# 第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
mu = -0.6
sigma = 0.15  # 将输出数据限制到0-1之间
diff1=[]
for i in range(10):
    diff1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])
# 第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
alpha = 1
beta = 10
diff2=[]
for i in range(10):
    diff2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])
X = torch.Tensor(diff1)
Y = torch.Tensor(diff2)
X,Y = Variable(X), Variable(Y)
mmd_rbf(X,Y)