"""
Author:wucng
Time:  20200110
Summary: 朴素贝叶斯对iris数据分类
源代码： https://github.com/wucng/MLAndDL
参考：https://cuijiahua.com/blog/2017/11/ml_4_bayes_1.html
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,auc
import pandas as pd
import numpy as np
from functools import reduce
from collections import Counter
import pickle,os,time

# 1.加载数据
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

class NaiveBayesClassifier(object):
    def __init__(self,save_file="model.ckpt"):
        self.save_file = save_file

    def fit(self,X:np.array,y:np.array):
        if not os.path.exists(self.save_file):
            # 计算分成每个类别的概率值
            dict_y = dict(Counter(y))
            dict_y = {k:v/len(y) for k,v in dict_y.items()}

            # 计算每维特征每个特征值发生概率值
            unique_label = list(set(y))
            dict_feature_value={} # 每个特征每个值对应的概率

            for col in range(len(X[0])):
                data = X[...,col] # 每列特征
                unique_val = list(set(data))
                for val in unique_val:
                    dict_feature_value[str(col)+"_"+str(val)] = np.sum(data==val)/len(data)

            dict_feature_value_label = {}  # 每个类别发生对应的每个特征每个值的概率
            for label in unique_label:
                datas = X[y==label]
                for col in range(len(datas[0])):
                    data = datas[..., col]  # 每列特征
                    unique_val = list(set(data))
                    for val in unique_val:
                        dict_feature_value_label[str(label)+"_"+str(col)+"_"+str(val)]=np.sum(data==val)/len(data)

            # save
            result={"dict_y":dict_y,"dict_feature_value":dict_feature_value,
                    "dict_feature_value_label":dict_feature_value_label}

            pickle.dump(result,open(self.save_file,"wb"))
            # return dict_y,dict_feature_value,dict_feature_value_label

    def __predict(self,X:np.array):
        data = pickle.load(open(self.save_file,"rb"))
        dict_y, dict_feature_value, dict_feature_value_label = data["dict_y"],data["dict_feature_value"],\
                                                               data["dict_feature_value_label"]

        labels = sorted(list(dict_y.keys()))
        # 计算每条数据分成每个类别的概率值
        preds = np.zeros([len(X),len(labels)])
        for i,x in enumerate(X):
            for j,label in enumerate(labels):
                p1 = 1
                p2 = 1
                for col,val in enumerate(x):
                    p1*= dict_feature_value_label[str(label)+"_"+str(col)+"_"+str(val)] if str(label)+"_"+str(col)+"_"+str(val) \
                    in dict_feature_value_label else self.__weighted_average(str(label)+"_"+str(col)+"_"+str(val),dict_feature_value_label) # self.__fixed_value()
                    p2*= dict_feature_value[str(col)+"_"+str(val)] if str(col)+"_"+str(val) in dict_feature_value else \
                    self.__weighted_average(str(col)+"_"+str(val),dict_feature_value) # self.__fixed_value()

                preds[i,j] = p1*dict_y[label]/p2

        return preds

    def __fixed_value(self):
        return 1e-3
    def __weighted_average(self,key:str,data_dict:dict):
        """插值方式找到离该key对应的最近的data_dict中的key做距离加权平均"""
        tmp = key.split("_")
        value = float(tmp[-1])
        if len(tmp)==3:
            tmp_key = tmp[0]+"_"+tmp[1]+"_"
        else:
            tmp_key = tmp[0] + "_"

        # 找到相关的key
        # related_keys = []
        values = [value]
        for k in list(data_dict.keys()):
            if tmp_key in k:
                # related_keys.append(k)
                values.append(float(k.split("_")[-1]))
        # 做距离加权
        values = sorted(values)
        index = values.index(value)
        # 取其前一个和后一个做插值
        last = max(0,index-1)
        next = min(index+1,len(values)-1)

        if index==last or index==next:
            return self.__fixed_value()
        else:
            d1=abs(values[last] - value)
            d2=abs(values[next] - value)
            v1 = data_dict[tmp_key+str(values[last])]
            v2 = data_dict[tmp_key+str(values[next])]

            # 距离加权 y=e^(-x)
            return (np.log(d1)*v1+np.log(d2)*v2)/(np.log(d1)+np.log(d2))

    def predict_proba(self,X:np.array):
        return self.__predict(X)

    def predict(self,X:np.array):
        return np.argmax(self.__predict(X),-1)

    def accuracy(self,y_true:np.array,y_pred:np.array)->float:
        return round(np.sum(y_pred==y_true)/len(y_pred),5)


if __name__=="__main__":
    dataPath = "../../dataset/iris.data"
    X, y = loadData(dataPath)
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=40)

    start = time.time()
    clf = NaiveBayesClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("cost time:%.6f(s) acc:%.3f"%(time.time()-start,clf.accuracy(y_test,y_pred)))
    # cost time:0.012998(s) acc:1.000

    # 使用sklearn 的GaussianNB
    start = time.time()
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("cost time:%.6f(s) acc:%.3f" % (time.time() - start, accuracy_score(y_test, y_pred)))
    # cost time:0.000996(s) acc:1.000