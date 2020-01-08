"""
Author:wucng
Time:  20200107
Summary: 使用KNN(K-近领域)对iris数据分类
数据下载：https://archive.ics.uci.edu/ml/datasets.php
源代码： https://github.com/wucng/MLAndDL
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import accuracy_score,auc
import pandas as pd
import numpy as np
import os
import time


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

class KNN(object):
    """默认使用欧式距离"""
    def __init__(self,X_train:np.asarray,X_test:np.asarray,
                 y_train:np.asarray,y_test:np.asarray=None,k:int=5):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.k = k

        self.__calDistance()

    # 2.计算每个测试样本到训练样本的距离
    def __calDistance(self):
        result_dist = np.zeros([len(self.X_test),len(self.X_train)])
        for i,data in enumerate(self.X_test):
            data = np.tile(data,(len(self.X_train),1))
            distance = np.sqrt(np.sum((data-self.X_train)**2,-1))
            # result_dist[i] = sorted(distance) # 从小到大排序,先不排序，否则索引位置发生变化与标签对应不上
            result_dist[i] = distance

        self.result_dist = result_dist
        # return result_dist

    def __calDistance2(self):
        result_dist = np.zeros([len(self.X_test), len(self.X_train)])
        for i, data_test in enumerate(self.X_test):
            dist = np.zeros((len(self.X_train),))
            for j,data in enumerate(self.X_train):
                dist[j] = (sum((data_test-data)**2))**0.5

            result_dist[i] = dist

        self.result_dist = result_dist
        # return result_dist

    # 3.根据距离确定类别
    def predict(self):
        """k:为选取的最近样本点个数"""
        # 距离从小到大排序获取索引
        result_index = np.argsort(self.result_dist,-1)[:,:self.k]

        # 将索引替换成对应的标签
        y_pred = self.y_train[result_index]

        # 统计每列次数出现最多对应的值即为预测标签
        y_pred = [np.bincount(pred).argmax() for pred in y_pred]
        self.y_pred = np.asarray(y_pred)

        return self.y_pred

    # 4.计算精度信息
    def accuracy(self):
        assert self.y_test is not None,print("error")
        assert len(self.y_pred)==len(self.y_test),print("error")
        return np.sum(self.y_pred==self.y_test)/len(self.y_test)

if __name__ =="__main__":
    dataPath = "../../dataset/iris.data"
    X,y = loadData(dataPath)
    # print(X.shape,y.shape) # (150, 4) (150,)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size = 0.2, random_state = 42)

    start = time.time()
    clf = KNN(X_train, X_test, y_train, y_test,3)
    y_pred = clf.predict()
    # print(y_pred)

    print("cost time:%.6f(s) acc:%.3f"%(time.time()-start,clf.accuracy()))
    # cost time:0.001994(s) acc:1.000

    # ----------------------------------------------------------------------
    # 使用sklearn的KNeighborsClassifier方法
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='kd_tree').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print("cost time:%.6f(s) error:%.3f" % (time.time() - start, acc))
    # cost time:0.001994(s) error:1.000