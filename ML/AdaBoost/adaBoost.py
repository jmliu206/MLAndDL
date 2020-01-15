"""
Author:wucng
Time:  20200115
Summary: 模型融合-Adaboost
源代码： https://github.com/wucng/MLAndDL
参考：https://cuijiahua.com/blog/2017/11/ml_10_adaboost.html
"""

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import scipy,pickle,os,time
import pandas as pd

# 1.加载数据集（并做预处理）
def loadData(dataPath: str) -> tuple:
    # 如果有标题可以省略header，names ；sep 为数据分割符
    df = pd.read_csv(dataPath, sep=",")
    # 填充缺失值
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    # df = df.fillna(0)
    # 数据量化
    # 文本量化
    df.replace("male", 0, inplace=True)
    df.replace("female", 1, inplace=True)

    df.loc[df["Embarked"] == "S", "Embarked"] = 0
    df.loc[df["Embarked"] == "C", "Embarked"] = 1
    df.loc[df["Embarked"] == "Q", "Embarked"] = 2

    # 划分出特征数据与标签数据
    X = df.drop(["PassengerId","Survived","Name","Ticket","Cabin"], axis=1)  # 特征数据
    y = df.Survived  # or df["Survived"] # 标签数据

    # 数据归一化
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    # 使用sklearn方式
    # X = MinMaxScaler().transform(X)

    # 查看df信息
    # df.info()
    # df.describe()
    return (X.to_numpy(), y.to_numpy())


class AdaboostClassifier(object):
    def __init__(self,modelList=[]):
        self.modelList = modelList
    def fit(self,X:np.array,y:np.array):
        # 样本权重，初始化成相等值，设定每个样本的权重都是相等的，即1/n
        D = np.ones((len(y)))*1/len(y)
        alpha_list =[]
        for model in self.modelList:
            model.fit(X, y,sample_weight=D) # 加上每个样本对应的权重
            # 计算错误率
            y_pred = model.predict(X)
            err = np.sum(y_pred!=y)/len(y)
            # 计算弱学习算法权重
            alpha = 1/2*np.log((1-err)/err)
            alpha_list.append(alpha)
            # 更新样本权重
            Z = np.sum(D)
            D = np.asarray([d/Z*np.exp(alpha*(-1)**(y_pred[i]==y[i])) for i,d in enumerate(D)])

        self.alpha_list=alpha_list

    def predict(self,X:np.array):
        preds = []
        last_preds = []
        for model,alpha in zip(self.modelList,self.alpha_list):
            y_pred = model.predict(X)
            preds.append(y_pred)
            last_preds.append(y_pred*alpha)

        y_pred = np.sign(np.sum(last_preds,0))
        return preds,y_pred


if __name__=="__main__":
    dataPath = "../../dataset/titannic/train.csv"
    X, y = loadData(dataPath)
    y[y==0]=-1 # 转成{-1,+1}标记，不使用{0,1}标记

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

    modelList = []
    for i in range(5):
        modelList.append(DecisionTreeClassifier("entropy", "random",random_state=i))
    # modelList.append(DecisionTreeClassifier(random_state=9))
    # modelList.append(DecisionTreeClassifier(splitter="random", random_state=0))
    # modelList.append(DecisionTreeClassifier("entropy", random_state=9))
    # modelList.append(DecisionTreeClassifier("entropy", "random", random_state=0))
    # modelList.append(DecisionTreeClassifier("entropy", "random", random_state=12))

    # modelList.append(LogisticRegression(penalty='l2', max_iter=1000, C=1.5))
    # modelList.append(LogisticRegression(penalty='l1', max_iter=1000, C=1.5))
    # modelList.append(LogisticRegression(penalty='l2', max_iter=2000, C=1.5))
    # modelList.append(LogisticRegression(penalty='l1', max_iter=5000, C=1.5))
    # modelList.append(LogisticRegression(penalty='l2', max_iter=5000, C=1.5))

    clf = AdaboostClassifier(modelList)
    clf.fit(X_train,y_train)
    preds,y_pred = clf.predict(X_test)

    # 计算acc
    # 每个单独的分类器的acc
    for i in range(len(modelList)):
        print("model:%d acc:%.5f" % (i, accuracy_score(preds[i], y_test)))
    # bagging的acc
    print("AdaBoost acc:%.5f" % (accuracy_score(y_pred, y_test)))

    # 使用sklearn 的 AdaBoostClassifier
    clf = AdaBoostClassifier(DecisionTreeClassifier(),10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("sklearn AdaBoost acc:%.5f" % (accuracy_score(y_pred, y_test)))