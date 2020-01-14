from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import scipy,pickle,os


class LinearRegressionSelf(object):
    def __init__(self,save_file="model.npy"):
        self.save_file = save_file

    def __fit(self,X,y):
        # 直接求导
        X = np.hstack((np.ones((len(X), 1)), X))
        # w = np.dot(np.linalg.inv(X),y) # 求逆
        w = np.dot(np.linalg.pinv(X), y)  # 求伪逆

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
        return np.dot(X,self.w)

    def error(self,y_true,y_pred):
        # https://www.jianshu.com/p/3a98f33113ac
        # 越大，拟合的效果越好,最优值为1，并且模型的效果很离谱时可能为负
        return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)

class LinearRegressionSelf2(object):
    """梯度下降"""
    def __init__(self,save_file="model.ckpt"):
        self.save_file = save_file

    def __fit(self,X,y,w,b,lr=1e-3):
        diff = np.dot(X, w) + b - y
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

                print(w,b)

            # save parameter
            pickle.dump({"w":w,"b":b},open(self.save_file,"wb"))

        data = pickle.load(open(self.save_file,"rb"))
        self.w = data["w"]
        self.b = data["b"]

    def predict(self,X):
        return np.dot(X,self.w)+self.b

    def error(self,y_true,y_pred):
        # https://www.jianshu.com/p/3a98f33113ac
        # 越大，拟合的效果越好,最优值为1，并且模型的效果很离谱时可能为负
        return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)

if __name__=="__main__":
    X, y = make_regression(200, 1, bias=2, noise=4,random_state=9)
    if len(y.shape)==1:y=y[...,None]
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    clf = LinearRegressionSelf()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("error:",clf.error(y_test,y_pred))
    # 画图
    plt.scatter(X_test,y_test,s=30,c='red',marker='o',alpha=0.5,label='C1')
    plt.plot(X_test,y_pred,c="blue")

    plt.show()