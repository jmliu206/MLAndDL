"""
Author:wucng
Time:  20200109
Summary: PCA 基于特征值分解
源代码： https://github.com/wucng/MLAndDL
"""
import scipy
# from scipy.misc import imread,imshow
from imageio import imread,imsave
import numpy as np
from numpy.linalg import eig,svd
import matplotlib.pyplot as plt
from sklearn.decomposition  import PCA

class PCASelf(object):
    def __init__(self,n_components=1,mode="svd"):
        """
        :param n_components:  主成分数（压缩后的特征数）
        :param mode: "eig" 特征矩阵分解，"svd" 奇异值分解
        """
        self.n_components = n_components
        self.mode = mode

    def fit_transform(self,X:np.array):
        # 去平均值
        X = X-np.mean(X,0)
        # 协方差矩阵
        A = np.matmul(X.T,X)#/len(X)

        if self.mode == "eig":
            # 计算协方差矩阵的特征值与特征向量
            vals, vecs = eig(A)
        else:
            # u, s, vh = np.linalg.svd(A)    # 这里vh为V的转置
            _, vals, vecs = svd(A)

        # 对特征值从大到小排序，选择其中最大的k个
        index = np.argsort(vals*(-1))[:self.n_components] # 默认是从小到大排序，乘上-1 后就变成从大到小排序

        # 根据选择的K个特征值组成新的特征向量矩阵（列对应特征向量，而不是行）
        P = vecs[:,index]

        # 特征压缩后的矩阵
        return np.matmul(X,P)

if __name__=="__main__":
    # 自定义方法
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCASelf(1)
    print(pca.fit_transform(X))

    # sklearn 方法
    print(PCA(1).fit_transform(X))