"""
计算数据集的经验熵和如何选择最优特征作为分类特征
https://github.com/Jack-Cherish/Machine-Learning
https://cuijiahua.com/blog/2017/11/ml_2_decision_tree_1.html
https://blog.csdn.net/c406495762/article/details/75663451
"""
from math import log2
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pickle,os

def createDataSet(dataPath):
    """
    # 文件是使用tab（"\t"）分割，因此sep="\t"  默认sep="," 如果是空格分割 则sep=" "
    # header = -1 没有标题行，然后使用names=[]重新设置标题
    df = pd.read_csv("../data/iris.data",header=-1,
                       names=["sepal_length","sepal_width","petal_length","petal_width","label"])
    
    # 缺失值处理
    df = df.fillna(0) # 将缺失值填充为0，也可以填充中值，平均值(计算缺失值所在的特征列的平均值或中值)
    # 或 df.dropna() 丢弃掉缺失值所在的行
    
    # 文本量化
    df.replace("Iris-setosa",0,inplace=True)
    df.replace("Iris-versicolor",1,inplace=True)
    df.replace("Iris-virginica",2,inplace=True)
    
    # 划分出特征数据与标签数据
    X = df.drop("label",axis=1) # 特征数据
    y = df.label # or df["label"] # 标签数据
    """
    df = pd.read_csv(dataPath,sep="\t") # header=0默认参数
    df = df.fillna(0)
    # 文本；量化
    df.replace("青年",0,inplace=True)
    df.replace("中年",1,inplace=True)
    df.replace("老年",2,inplace=True)

    df.replace("否",0,inplace=True)
    df.replace("是",1,inplace=True)

    df.replace("一般",1,inplace=True)
    df.replace("好",2,inplace=True)
    df.replace("非常好",3,inplace=True)

    # 去掉多出的ID列 不属于特征列和标签列
    df = df.drop("ID",axis=1)

    # 转成numpy做后续计算
    data = df.to_numpy()

    return data,list(df.columns)

# 手动解析数据
def createDataSet2(dataPath,hasTitle=True):
    features = []
    with open(dataPath,"r",encoding="utf-8") as fp:
        lines = fp.readlines()
        for line in lines: # 一行一行解析
            if hasTitle: # 跳过第一行 标题行
                hasTitle = False
                continue
            tmp = []
            data = line.strip().split("\t")
            for da in data[1:]: # 去掉ID列，如果没有 则为 data
                if da =="青年":
                    tmp.append(0)
                elif da =="中年":
                    tmp.append(1)
                elif da =="老年":
                    tmp.append(2)

                elif da == "否":
                    tmp.append(0)
                elif da == "是":
                    tmp.append(1)

                elif da == "一般":
                    tmp.append(1)
                elif da == "好":
                    tmp.append(2)
                elif da == "非常好":
                    tmp.append(3)

            features.append(tmp)

    return np.asarray(features,np.int32)


# 计算数据集的信息熵(香农熵)
"""
H = - (p(x_1)log2(p(x_1))+p(x_2)log2(p(x_2))+...+p(x_n)log2(p(x_n)))
n是分类的数目。熵越大，随机变量的不确定性就越大
这里使用频率代替概率得到是经验熵
"""
def calEmpiricalEntropy(data):
    label = data[:,-1] # 最后一列为标签列
    # unique_label_value = set(label) # 去重 得到唯一值
    nums_data = len(label)
    entropy = 0.0 # 熵
    dict_label = dict(Counter(label)) # 计算每个类别数量  {0: 6, 1: 9}
    for k,v in dict_label.items():
        entropy+=v/nums_data*log2(v/nums_data)

    entropy *=(-1) # 取相反
    return entropy

# 计算条件熵（经验条件熵）
def calConditionalEntropy(data,featureColumn=0):
    # 计算某列特征的条件熵
    nums_data = len(data)
    oneFeature = data[:,featureColumn] # 获取这列特征
    labels = data[:,-1] # 最后一列为标签列
    Di = {}
    Dik= {}
    for value,label in zip(oneFeature,labels):
        # 统计特征值为某个值时对应的样本数
        if value not in Di:
            Di[value]=0
        Di[value] += 1

        # 统计特征值为某个值时，不同label对应的数量
        key = str(value)+"_"+str(label)
        if key not in Dik:
            Dik[key]=0
        Dik[key] += 1

    # 计算条件熵
    entropy = 0
    for k,value in Di.items():
        tem_entropy = 0
        for label in set(labels):
            key = str(k)+"_"+str(label)
            if key in Dik:
                tem_entropy += Dik[key]/value*log2(Dik[key]/value)
        entropy += value/nums_data*tem_entropy

    entropy *=(-1) # 记得取反

    return entropy

# 根据信息增益选择最好的特征分裂(信息增益越大 其实也是 条件熵越小)
def chooseBestFeatureToSplit(data):
    numFeatures = len(data[0]) - 1 # 列数,减去一列label列
    baseEntropy = calEmpiricalEntropy(data)
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):
        condEntropy = calConditionalEntropy(data,i)
        infoGain = baseEntropy - condEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
        # print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值降序排序
    return sortedClassCount[0][0]

def splitDataSet(dataSet,bestFeat,value):
    dataSet = dataSet[dataSet[:,bestFeat] == value]
    return np.hstack((dataSet[:,0:bestFeat],dataSet[:,bestFeat+1:]))

# 创建决策树
def createTree(dataSet, labels, featLabels):
    classList = dataSet[:,-1].tolist()           #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:                                    #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优特征
    bestFeatLabel = labels[bestFeat]                            #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                                    #根据最优特征的标签生成树
    del(labels[bestFeat])                                        #删除已经使用特征标签
    featValues = dataSet[:,bestFeat]        #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                   #遍历特征，创建决策树。
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value), subLabels, featLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))                                                        #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

def getFeatLabels(inputTree,featLabels = []):
    firstStr = next(iter(inputTree))
    featLabels.append(firstStr)
    secondDict = inputTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            featLabels = getFeatLabels(secondDict[key],featLabels)
    return featLabels

if __name__ == '__main__':
    filename = "model.ckpt"
    if not os.path.exists(filename):
        dataPath = "../data/loan.data"
        data,column_names= createDataSet(dataPath)
        # data = createDataSet2(dataPath)
        # calEmpiricalEntropy(data[:,-1]) # 0.9709505944546686

        # print("最优特征索引值:" + str(chooseBestFeatureToSplit(data)))
        """
        第0个特征的增益为0.083
        第1个特征的增益为0.324
        第2个特征的增益为0.420
        第3个特征的增益为0.363
        最优特征索引值:2
        """

        # 按i=2列将数据集D拆成D1,D2数据集
        # D2 = data[data[:, 2] == 0]
        # print("最优特征索引值:" + str(chooseBestFeatureToSplit(D2)))
        """
        第0个特征的增益为0.252
        第1个特征的增益为0.918
        第2个特征的增益为0.000
        第3个特征的增益为0.474
        最优特征索引值:1
        """
        featLabels = []
        myTree =createTree(data,column_names[:-1],featLabels)

        # 保存创建好的决策树
        pickle.dump(myTree,open(filename, 'wb'))

        print(myTree)
    else:
        # 加载模型
        myTree = pickle.load(open(filename, 'rb'))
        featLabels = []
        featLabels = getFeatLabels(myTree, featLabels=featLabels)

        testVec = [0, 1]  # 测试数据
        result = classify(myTree, featLabels, testVec)
        if result == 1:
            print('放贷')
        if result == 0:
            print('不放贷')


