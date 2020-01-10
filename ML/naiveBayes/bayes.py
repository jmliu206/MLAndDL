"""
Author:wucng
Time:  20200109
Summary: 朴素贝叶斯分类
源代码： https://github.com/wucng/MLAndDL
参考：https://cuijiahua.com/blog/2017/11/ml_4_bayes_1.html
"""

import numpy as np
from functools import reduce
from collections import Counter
import pickle,os

# 1.加载数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

# 2.文本向量化
def createVocabList(dataSet:list)->list:
    # 先建立词汇表
    vocabSet = set([])
    for data in dataSet:
        vocabSet = vocabSet | set(data) # 并集

    return sorted(list(vocabSet))

def word2vec(vocabList,dataSet):
    vecs = np.zeros([len(dataSet),len(vocabList)])
    for i,data in enumerate(dataSet):
        for word in data:
            if word in vocabList:
                vecs[i,vocabList.index(word)] = 1 # 标记为1
    return vecs

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
                    in dict_feature_value_label else self.__weighted_average(str(label)+"_"+str(col)+"_"+str(val),dict_feature_value_label)
                    p2*= dict_feature_value[str(col)+"_"+str(val)] if str(col)+"_"+str(val) in dict_feature_value else \
                    self.__weighted_average(str(col)+"_"+str(val),dict_feature_value)


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
    dataset,label = loadDataSet()
    vocabList = createVocabList(dataset)
    dataset = word2vec(vocabList,dataset)
    label = np.asarray(label)
    # print(dataset.shape,label.shape) # (6, 32) (6,)

    clf = NaiveBayesClassifier()
    clf.fit(dataset,label)
    print(clf.predict(dataset))