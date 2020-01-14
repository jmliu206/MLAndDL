"""
Author:wucng
Time:  20200114
Summary: 逻辑回归对iris数据分类
源代码： https://github.com/wucng/MLAndDL
参考：https://www.jianshu.com/p/ba60f232e9da
"""

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import scipy,pickle,os,time
import pandas as pd

# 1.加载数据集（并做预处理）
def loadData(dataPath: str) -> tuple:
    # 如果有标题可以省略header，names ；sep 为数据分割符
    df = pd.read_csv(dataPath, sep=",", header=-1,
                     names=["sepal_length", "sepal_width", "petal_length", "petal_width", "label"])
    # 填充缺失值
    df = df.fillna(0)
    # 数据量化
    # 文本量化
    df.replace("Iris-setosa", 0, inplace=True) # 0
    df.replace("Iris-versicolor", 1, inplace=True) # 1
    df.replace("Iris-virginica", 2, inplace=True)

    X = df.drop("label", axis=1)  # 特征数据
    y = df.label  # or df["label"] # 标签数据

    # 数据归一化
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    return (X.to_numpy()[:100], y.to_numpy()[:100])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_inv(y):
    # 如果y取0,1 做标签，会出现正无穷或负无穷，导致w求出都为nan
    # 此时如果使用softlabel，即 0 取0.2 ，1取0.8，可以很好的求出参数w
    return np.log(y)-np.log(1-y)

class LogisticRegressionSelf(object):
    """
    直接使用求伪逆
    g(Xw)=y => Xw=g_inv(y) => w = pinv(X)*g_inv(y)
    """
    def __init__(self,save_file="model.npy"):
        self.save_file = save_file

    def __fit(self,X,y):
        # 直接求导
        X = np.hstack((np.ones((len(X), 1)), X))
        w = np.dot(np.linalg.pinv(X), sigmoid_inv(y))  # 求伪逆
        return w

    def fit(self,X,y,batch_size=32,epochs=1):
        if not os.path.exists(self.save_file):
            length = len(y)
            m = len(y)//batch_size
            last_w = []
            for epoch in range(epochs):
                w = []
                # 随机打乱数据
                index = np.arange(0, length)
                np.random.seed(epoch)
                np.random.shuffle(index)
                new_X = X[index]
                new_y = y[index]
                for i in range(m):
                    start = i*batch_size
                    end = min((i+1)*batch_size,length)
                    w.append(self.__fit(new_X[start:end],new_y[start:end]))

                last_w.append(np.mean(w,0))

            # save parameter
            np.save(self.save_file,np.mean(last_w,0))

        self.w = np.load(self.save_file)

    def predict(self,X):
        X = np.hstack((np.ones((len(X), 1)), X))
        return (sigmoid(np.dot(X,self.w))>0.5).astype(np.float32)

    def accuracy(self,y_true,y_pred):
        return round(np.sum(y_pred==y_true)/len(y_true),5)

class LogisticRegressionSelf2(object):
    """梯度下降"""
    def __init__(self,save_file="model.ckpt"):
        self.save_file = save_file

    def __fit(self,X,y,w,b,lr=1e-3):
        diff = sigmoid(np.dot(X, w) + b) - y
        w-=lr*(1/len(y))*(np.dot(np.transpose(X), diff))
        b-=lr*np.mean(diff)

        return w,b

    def fit(self,X,y,batch_size=32,epochs=5000,lr=5e-4):
        if not os.path.exists(self.save_file):
            length = len(y)
            m = len(y)//batch_size
            w = np.random.random((len(X[0]),1)) # 初始随机值
            b = np.random.random((1,1)) # 初始随机值

            for epoch in range(epochs):
                # 随机打乱数据
                index = np.arange(0, length)
                np.random.seed(epoch)
                np.random.shuffle(index)
                new_X = X[index]
                new_y = y[index]
                for i in range(m):
                    start = i*batch_size
                    end = min((i+1)*batch_size,length)
                    w,b = self.__fit(new_X[start:end],new_y[start:end],w,b,lr)

                # print(w,b)

            # save parameter
            pickle.dump({"w":w,"b":b},open(self.save_file,"wb"))

        data = pickle.load(open(self.save_file,"rb"))
        self.w = data["w"]
        self.b = data["b"]

    def predict(self,X):
        return (sigmoid(np.dot(X,self.w)+self.b)>0.5).astype(float)

    def accuracy(self, y_true, y_pred):
        return round(np.sum(y_pred == y_true) / len(y_true), 5)

if __name__=="__main__":
    dataPath = "../../dataset/iris.data"
    X, y = loadData(dataPath)
    # 转成soft label
    y[y==0]=0.2
    y[y==1]=0.8
    if len(y.shape)==1:y=y[...,None]
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    # 矩阵求逆方式
    start =time.time()
    clf = LogisticRegressionSelf()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("cost time:%.5f acc:%.5f"%(time.time()-start,clf.accuracy((y_test > 0.5).astype(float), y_pred)))

    # ----------------------------------------------------------
    X, y = loadData(dataPath)
    if len(y.shape)==1:y=y[...,None]
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    # 梯度下降方式
    start = time.time()
    clf = LogisticRegressionSelf2()
    clf.fit(X_train,y_train,batch_size=16,epochs=1000,lr=1e-3)
    y_pred = clf.predict(X_test)
    print("cost time:%.5f acc:%.5f"%(time.time()-start,clf.accuracy(y_test, y_pred)))

    # sklearn 的LogisticRegression
    start = time.time()
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("cost time:%.5f acc:%.5f" % (time.time() - start, accuracy_score(y_test,y_pred)))