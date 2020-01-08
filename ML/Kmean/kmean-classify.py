"""
Author:wucng
Time:  20200108
Summary: 使用Kmean算法对iris数据分类 （数据有标签）
数据下载：https://archive.ics.uci.edu/ml/datasets.php
源代码： https://github.com/wucng/MLAndDL
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import accuracy_score,auc
import pandas as pd
import numpy as np
import os
import time
import pickle

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

class KMeanClassifier():
    """默认使用欧式距离"""
    def __init__(self, X_train: np.asarray, y_train: np.asarray,
                  savefile="./model.ckpt"):
        self.X_train = X_train
        self.y_train = y_train
        self.savefile = savefile
        if not os.path.exists(savefile):
            self.__calClassCenter()
        self.data = pickle.load(open(self.savefile,"rb"))

    # 2.训练样本按标签聚类，计算每个类的中心
    def __calClassCenter(self):
        # 按类别建立一个dict
        dataset={}
        for x,y in zip(self.X_train,self.y_train):
            if y not in dataset:
                dataset[y]=[]
            dataset[y].append(x)

        # 计算每个类别的中心
        data = {}
        center = []
        labels = []
        for label in dataset:
            # data[label]=np.mean(np.asarray(dataset[label]),0)
            labels.append(label)
            center.append(np.mean(np.asarray(dataset[label]),0))
            # center.append(np.median(np.asarray(dataset[label]),0))

        data["label"] = labels
        data["center"] = center

        # 将这个dict保存，下次就可以不用再重新建立(节省时间)
        pickle.dump(data,open(self.savefile,"wb"))
        # return data

    # 3.预测样本
    def predict(self,X_test: np.asarray)->np.asarray:
        labels = np.asarray(self.data["label"])
        center = np.asarray(self.data["center"])
        result_dist = np.zeros([len(X_test), len(center)])
        for i, data in enumerate(X_test):
            data = np.tile(data, (len(center), 1))
            distance = np.sqrt(np.sum((data - center) ** 2, -1))
            result_dist[i] = distance

        # 距离从小到大排序获取索引
        result_index = np.argsort(result_dist, -1)

        # 将索引替换成对应的标签，取距离最小对应的类别
        y_pred = labels[result_index][...,0]

        return y_pred

    # 4.计算精度信息
    def accuracy(self,y_true,y_pred)->float:
        return round(np.sum(y_pred == y_true) / len(y_pred),5)


if __name__ =="__main__":
    dataPath = "../../dataset/iris.data"
    X,y = loadData(dataPath)
    # print(X.shape,y.shape) # (150, 4) (150,)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size = 0.2, random_state = 42)

    start = time.time()
    clf = KMeanClassifier(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc = clf.accuracy(y_test,y_pred)

    print("cost time:%.6f(s) acc:%.3f" % (time.time() - start, acc))
    # cost time:0.000984(s) acc:0.967