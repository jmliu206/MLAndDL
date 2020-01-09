"""
Author:wucng
Time:  20200109
Summary: 使用Kmean算法实现图像压缩
源代码： https://github.com/wucng/MLAndDL
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import time

class KMeanCompress(object):
    def __init__(self,num_samples,n_clusters=8,random_state=None):
        self.num_samples = num_samples # 采样数量，选择多少条来建立KMean模型
        self.n_clusters = n_clusters
        self.random_state= random_state

    def fit(self,X:np.array):
        np.random.seed(self.random_state)
        data = X.copy()
        np.random.shuffle(data)
        data = data[...,:self.num_samples]

        # Kmean建模
        self.kmean = KMeans(n_clusters=self.n_clusters,random_state=self.random_state).fit(data)

    def predict(self,X:np.array):
        self.labels = self.kmean.predict(X)

    def compress(self,h:int,w:int,c:int)->np.array:
        new_img = np.zeros([h,w,c])
        for i in range(h):
            for j in range(w):
                index = j + i*w
                new_img[i,j,:] = self.kmean.cluster_centers_[self.labels[index]]

        return new_img


if __name__=="__main__":
    img = Image.open("../../dataset/test.jpg").convert("RGB").resize((224,224))
    img.show()
    img = np.array(img)/255.
    h,w,c = img.shape
    img_arr = np.reshape(img, (h*w, c))

    clf = KMeanCompress(500,64,9)
    clf.fit(img_arr)
    clf.predict(img_arr)
    new_img = clf.compress(h,w,c)

    Image.fromarray(np.clip(new_img*255,0,255).astype(np.uint8)).show()