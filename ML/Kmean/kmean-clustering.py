"""
Author:wucng
Time:  20200108
Summary: 使用Kmean算法对iris数据聚类
数据下载：https://archive.ics.uci.edu/ml/datasets.php
源代码： https://github.com/wucng/MLAndDL
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,auc
import pandas as pd
import numpy as np
import os
import time
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1.加载数据集（并做预处理）
def loadData(dataPath: str) -> tuple:
    # 如果有标题可以省略header，names ；sep 为数据分割符
    df = pd.read_csv(dataPath, sep=",", header=-1,
                     names=["sepal_length", "sepal_width", "petal_length", "petal_width", "label"])
    # 填充缺失值
    df = df.fillna(0)
    # 数据量化
    # 文本量化
    df.replace("Iris-setosa", 0, inplace=True)
    df.replace("Iris-versicolor", 1, inplace=True)
    df.replace("Iris-virginica", 2, inplace=True)

    # 划分出特征数据与标签数据
    X = df.drop("label", axis=1)  # 特征数据
    y = df.label  # or df["label"] # 标签数据

    # 数据归一化
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    # 使用sklearn方式
    # X = MinMaxScaler().transform(X)

    # 查看df信息
    # df.info()
    # df.describe()
    return (X.to_numpy(), y.to_numpy())

class KMeanCluster():
    """默认使用欧式距离"""
    def __init__(self,n_clusters=3, max_iter=300, error=1e-4, random_state=None,
                 savefile="./center.npy"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.savefile = savefile

    def __calClassCenter(self,centers:np.asarray,X:np.asarray,isReturnPred:bool=False)->np.asarray:
        labels = np.arange(0,len(centers))
        result_dist = np.zeros([len(X), len(centers)])
        for i, data in enumerate(X):
            data = np.tile(data, (len(centers), 1))
            distance = np.sqrt(np.sum((data - centers) ** 2, -1))
            result_dist[i] = distance

        # 距离从小到大排序获取索引
        result_index = np.argsort(result_dist, -1)

        # 将索引替换成对应的标签，取距离最小对应的类别
        y_pred = labels[result_index][..., 0]

        if isReturnPred:
            return y_pred

        # 按类别建立一个dict
        dataset = {}
        for x,y in zip(X,y_pred):
            if y not in dataset:
                dataset[y] = []
            dataset[y].append(x)

        # 计算每个类别的中心
        center = []
        for label in labels:
            center.append(np.mean(np.asarray(dataset[label]), 0))

        return np.asarray(center)

    # 构建聚类
    def __fit_transform(self, X, y=None, sample_weight=None):
        # 1.随机选择聚类中心
        random.seed(self.random_state)
        centers = np.asarray(random.choices(X,k=self.n_clusters))

        # tqdm_bar = tqdm(range(self.max_iter))
        # for i in tqdm_bar:
        for i in range(self.max_iter):
            # 2.根据聚类中心计算每个样本属于哪个聚类,再更新聚类中心
            new_centers = self.__calClassCenter(centers,X)

            # 计算新的聚类中心与原来的中心之间的误差
            error = np.sum((new_centers-centers)**2)/len(centers)

            print("step:%s\terror:%f\tmin_error:%f" % (i, error, self.error))

            if error > self.error:
                # 更新聚类中心
                centers = new_centers
            else: # 停止迭代
                break

            # tqdm_bar.set_description_str("step:%s\terror:%.5f"%(i,error))

        # self.centers = centers
        # 保存
        np.save(self.savefile,centers)
        # return centers

    def fit_transform(self, X, y=None, sample_weight=None):
        if not os.path.exists(self.savefile):
            self.__fit_transform(X)
        self.centers = np.load(self.savefile)

    def predict(self, X, sample_weight=None):
        return self.__calClassCenter(self.centers,X,True)


if __name__ =="__main__":
    dataPath = "../../dataset/iris.data"
    X,y = loadData(dataPath)
    # print(X.shape,y.shape) # (150, 4) (150,)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=0.2, random_state=42)

    clf = KMeanCluster(n_clusters=3, random_state=9,error=1e-7)
    clf.fit_transform(X)
    y_pred = clf.predict(X)

    plt.subplot(131)
    plt.scatter(X[:,0],X[:,2],c=y)
    # plt.legend(y.tolist(), loc = 'upper right')
    plt.title("origin")

    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 2], c=y_pred)
    # plt.legend(y_pred.tolist(), loc='upper right')
    plt.title("custom kmean")

    # -------------------------------------------------------------
    # sklearn的KMeans
    y_pred = KMeans(n_clusters=3, random_state=9,tol=1e-7).fit_predict(X)
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 2], c=y_pred)
    # plt.legend(y_pred.tolist(), loc='upper right')
    plt.title("sklearn kmean")

    plt.show()