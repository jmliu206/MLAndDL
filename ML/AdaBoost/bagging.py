"""
Author:wucng
Time:  20200115
Summary: 模型融合-Bagging
源代码： https://github.com/wucng/MLAndDL
参考：https://cuijiahua.com/blog/2017/11/ml_10_adaboost.html
"""

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier,VotingClassifier
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

class BaggingClassifierSelf(object):
    def __init__(self,modelList=[]):
        self.modelList = modelList

    def fit(self,X:np.array,y:np.array):
        index = np.arange(0,len(y),dtype = np.int32)
        for model in self.modelList:
            # 有放回的随机采样
            new_index = np.random.choice(index, size=len(y))
            new_X =X[new_index]
            new_y = y[new_index]

            model.fit(new_X,new_y)

    def predict(self,X:np.array):
        preds =[]
        for model in self.modelList:
            preds.append(model.predict(X))
        # 按少数服从多数（投票机制）
        new_preds = np.array(preds).T
        # 统计每列次数出现最多对应的值即为预测标签
        y_pred = [np.bincount(pred).argmax() for pred in new_preds]
        return preds,y_pred

    def predict_proba(self,X:np.array):
        preds_proba=[]
        for model in self.modelList:
            preds_proba.append(model.predict_proba(X))
        # 直接计算分成没类分数取平均
        y_preds_proba = np.zeros_like(preds_proba[0])
        for proba in preds_proba:
            y_preds_proba += proba

        # 取平均
        y_preds_proba = y_preds_proba/len(preds_proba)

        return preds_proba,y_preds_proba


if __name__=="__main__":
    dataPath = "../../dataset/titannic/train.csv"
    X, y = loadData(dataPath)
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

    # 创建5个决策树模型，也可以使用其他模型混合(模型不一定要一样)
    modelList =[]
    modelList.append(DecisionTreeClassifier(random_state=9))
    modelList.append(DecisionTreeClassifier(splitter="random",random_state=0))
    modelList.append(DecisionTreeClassifier("entropy",random_state=9))
    modelList.append(DecisionTreeClassifier("entropy","random",random_state=0))
    modelList.append(DecisionTreeClassifier("entropy","random",random_state=12))

    modelList.append(LogisticRegression(penalty='l2',max_iter=1000,C=1.5))
    modelList.append(LogisticRegression(penalty='l1',max_iter=1000,C=1.5))
    modelList.append(LogisticRegression(penalty='l2',max_iter=2000,C=1.5))
    modelList.append(LogisticRegression(penalty='l1', max_iter=5000, C=1.5))
    modelList.append(LogisticRegression(penalty='l2', max_iter=5000, C=1.5))

    clf_bagging = BaggingClassifierSelf(modelList)
    clf_bagging.fit(X_train,y_train)

    preds,y_pred = clf_bagging.predict(X_test)
    # 计算acc
    # 每个单独的分类器的acc
    for i in range(len(modelList)):
        print("model:%d acc:%.5f" % (i, accuracy_score(preds[i], y_test)))
    # bagging的acc
    print("bagging acc:%.5f" % (accuracy_score(y_pred, y_test)))

    print("-----------------------------------------------------")
    preds_proba, y_preds_proba = clf_bagging.predict_proba(X_test)
    preds, y_pred = [proba.argmax(axis=-1) for proba in preds_proba], y_preds_proba.argmax(axis=-1)

    # 计算acc
    # 每个单独的分类器的acc
    for i in range(len(modelList)):
        print("model:%d acc:%.5f"%(i,accuracy_score(preds[i],y_test)))
    # bagging的acc
    print("bagging acc:%.5f" % (accuracy_score(y_pred, y_test)))

    # sklearn 的 BaggingClassifier
    clf = BaggingClassifier(DecisionTreeClassifier())
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("sklearn bagging acc:%.5f" % (accuracy_score(y_pred, y_test)))

    # sklearn 的 VotingClassifier
    clf = VotingClassifier(estimators=[(str(i),model) for i,model in enumerate(modelList)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("sklearn bagging acc:%.5f" % (accuracy_score(y_pred, y_test)))