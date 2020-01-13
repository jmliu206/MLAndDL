"""
Author:wucng
Time:  20200113
Summary: 决策树对titanic数据分类
源代码： https://github.com/wucng/MLAndDL
参考：https://cuijiahua.com/blog/2017/11/ml_3_decision_tree_2.html
"""

from math import log2
import numpy as np
import pandas as pd
from collections import Counter
import operator
import pickle,os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def loadData(dataPath,sep=","):
    df = pd.read_csv(dataPath,sep=sep)
    df = df.drop("ID",axis=1)
    """
    # 文本量化
    df.loc[df["病症"] == "打喷嚏", "病症"] = 0
    df.loc[df["病症"] == "头痛", "病症"] = 1

    df.loc[df["职业"] == "护士", "职业"] = 0
    df.loc[df["职业"] == "农夫", "职业"] = 1
    df.loc[df["职业"] == "建筑工人", "职业"] = 2
    df.loc[df["职业"] == "教师", "职业"] = 3

    df.loc[df["疾病"] == "感冒", "疾病"] = 0
    df.loc[df["疾病"] == "过敏", "疾病"] = 1
    # df.loc[df["疾病"] == "脑震荡", "疾病"] = 2
    df.replace("脑震荡",2,inplace=True)
    """
    return df.to_numpy(),list(df.columns)[:-1]

# 1.加载数据集（并做预处理）
def loadData2(dataPath: str) -> tuple:
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
    return (X.to_numpy(), y.to_numpy(),list(X.columns))

class DecisonTreeClassifierSelf(object):
    def __init__(self,save_file="./model.ckpt"):
        self.save_file = save_file

    # 计算经验熵
    def __calEmpiricalEntropy(self,dataset):
        label = dataset[..., -1]
        label_dict = dict(Counter(label))
        entropy = 0
        for k, v in label_dict.items():
            p = v / len(label)
            entropy += p * log2(p)
        entropy *= -1
        return entropy

    # 条件熵
    def __calConditionalEntropy(self,dataset, feature_index=0):
        label = dataset[..., -1]
        select_feature = dataset[..., feature_index]
        uniqueValue = set(select_feature)

        conditionalEntropy = 0
        # 根据唯一值划分成子集
        for value in uniqueValue:
            select_rows = select_feature == value
            # subset_feature = select_feature[select_rows]
            subset_label = label[select_rows]

            conditionalEntropy += len(subset_label) / len(label) * self.__calEmpiricalEntropy(subset_label[..., None])

        # conditionalEntropy *= -1 # 计算calEmpiricalEntropy 已乘上了-1

        return conditionalEntropy

    # 根据信息增益选择最佳的特征进行分裂生长
    # 信息增益 = 经验熵-条件熵
    def __chooseBestFeatureToSplit(self,dataset):
        # 特征个数
        numFeatures = dataset.shape[1] - 1  # 有列是label
        baseEntropy = self.__calEmpiricalEntropy(dataset)
        bestInfoGain = - 9999
        bestFeature = -1

        for i in range(numFeatures):
            infoGain = baseEntropy - self.__calConditionalEntropy(dataset, i)
            # print("第%d个特征的增益为%.3f" % (i, infoGain))
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i

        return bestFeature

    def __majorityCnt(self,classList):
        classCount = dict(Counter(classList))
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
        return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素

    def __splitDataSet(self,dataSet, axis, value):
        retDataSet = []  # 创建返回的数据集列表
        for featVec in dataSet:  # 遍历数据集
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis].tolist()  # 去掉axis特征
                reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
                retDataSet.append(reducedFeatVec)
        return np.asarray(retDataSet)

    # 函数说明:创建决策树
    def __createTree(self,dataset, labels, featLabels):
        """
        如果完全分成1个类别则为叶结点，停止生长
        如果没有完全分成1个类别，但是没有可使用的特征，停止生长，并选择类别个数最多的作为叶结点
        """
        classList = dataset[..., -1].tolist()
        if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
            return classList[0]
        if len(dataset[0]) == 1 or len(labels) == 0:  # 遍历完所有特征时返回出现次数最多的类标签
            return self.__majorityCnt(classList)

        bestFeat = self.__chooseBestFeatureToSplit(dataset)  # 选择最优特征
        bestFeatLabel = labels[bestFeat]  # 最优特征的标签
        featLabels.append(bestFeatLabel)
        myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
        del (labels[bestFeat])  # 删除已经使用特征标签
        featValues = dataset[..., bestFeat]  # 得到训练集中所有最优特征的属性值
        uniqueVals = set(featValues)  # 去掉重复的属性值
        for value in uniqueVals:  # 遍历特征，创建决策树。
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.__createTree(self.__splitDataSet(dataset, bestFeat, value), subLabels, featLabels)
        return myTree

    def fit(self,dataset,feature_names):
        if not os.path.exists(self.save_file):
            featLabels = []
            myTree = self.__createTree(dataset, feature_names, featLabels)

            data_dict = {}

            data_dict["myTree"] = myTree
            # data_dict["featLabels"] = featLabels
            # data_dict["feature_names"] = feature_names

            # sava model
            pickle.dump(data_dict,open(self.save_file,"wb"))

        self.data_dict = pickle.load(open(self.save_file,"rb"))
        # return myTree

    def __predict(self, inputTree, featLabels, testVec):
        firstStr = next(iter(inputTree))  # 获取决策树结点
        secondDict = inputTree[firstStr]  # 下一个字典
        featIndex = featLabels.index(firstStr)
        # classLabel = 0
        for key in secondDict.keys():
            if testVec[featIndex] != key:
                key = testVec[featIndex]
                # 找离该值最近的key
                keys = list(secondDict.keys())
                keys.append(key)
                keys = sorted(keys)
                index = keys.index(key)
                if index ==0:
                    key = keys[index+1]
                elif index == len(keys)-1:
                    key = keys[index-1]
                else:
                    if abs(keys[index-1]-keys[index])<=abs(keys[index + 1]-keys[index]):
                        key = keys[index-1]
                    else:
                        key = keys[index + 1]

            if isinstance(secondDict[key], dict):
                classLabel = self.__predict(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

        return classLabel

    def predict(self,X:np.array,feature_names):
        # featLabels = self.data_dict["featLabels"]
        featLabels = feature_names
        inputTree = self.data_dict["myTree"]
        labels = []
        for x in X:
            labels.append(self.__predict(inputTree, featLabels, x))
        return np.asarray(labels)


    def accuracy(self,y_true,y_pred):
        return round(np.sum(y_pred==y_true)/len(y_true),5)


if __name__=="__main__":
    # dataPath = "../../dataset/medical_record.data"
    # dataPath = "../../dataset/loan.data"
    # dataset,feature_names = loadData(dataPath,sep="\t")
    # clf = DecisonTreeClassifierSelf()
    # clf.fit(dataset, feature_names)
    # y_pred = clf.predict(dataset[..., :-1], feature_names)
    # print(y_pred)
    #
    # print(clf.accuracy(dataset[..., -1], y_pred))
    dataPath = "../../dataset/titannic/train.csv"
    X,y,feature_names = loadData2(dataPath)
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
    clf = DecisonTreeClassifierSelf()
    clf.fit(np.hstack((X_train,y_train[...,None])), feature_names.copy())
    y_pred = clf.predict(X_test,feature_names)
    print(clf.accuracy(y_test, y_pred))
    # 0.74444
    # for y1,y2 in zip(y_pred,y_test):
    #     print(y1,"\t",y2)

    # sklearn 的 DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test,y_pred))
    # 0.77777