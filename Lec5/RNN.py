#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

##### データの取得
#クラス数を定義
m = 4

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train[y_train < m, :, :]

x_test = x_test.astype('float32') / 255.
x_test = x_test[y_test < m, :, :]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d , _ = x_train.shape
n_test, _, _ = x_test.shape

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播
def ReLU(x):
    return np.maximum(0,x),np.where(x>0,1,0)

def sigmoid(x):
    ret=1/(1+np.exp(-x))
    return ret,(1-ret)*ret

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def CrossEntoropy(x, y):
    return -np.sum(y*np.log(x))

def forward(x,z,w_in,wh,fncs):
    #x=x.reshape([29,1])
    z_new,a=fncs(np.dot(w_in,x)+np.dot(wh,z))
    #b=np.append(1,z_new.T)
    return z_new,a

def backward(w_hidden,delta,w_out,delta_out,derivative):
    w_out=np.delete(w_out,0,1)
    return (np.dot(w_hidden.T,delta)+np.dot(w_out.T,delta_out))*derivative
    
##### 中間層のユニット数とパラメータの初期値
q = 200
w_hidden = np.random.normal(0, 0.3, size=(q,q))
w_in = np.random.normal(0, 0.3, size=(q, d+1))     #W_inの定義
w_out = np.random.normal(0, 0.3, size=(m, q+1)) #中間層から出力層へのパラメータ


########## 誤差逆伝播法によるパラメータ推定
num_epoch = 50
eta = 10**(-3)

e = []
e_test = []
error = []
error_test = []

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    print(epoch)
    eta_t = eta/(epoch + 1) #学習率

    for i in index:
        xi = x_train[i, :, :]
        yi = y_train[i, :]
        ##### 順伝播
        Z=np.zeros((d+1,q))
        U=np.zeros((d,q))
        z_out=np.zeros((d,m))
        for j in range(1,d+1):
            Z[j], U[j-1] = forward(np.append(1, xi[j-1]), Z[j-1], w_in, w_hidden, sigmoid)
            z_out[j-1] = softmax(np.dot(w_out, np.append(1, Z[j])))
        ##### 誤差評価: 誤差をeにappendする
        e.append(CrossEntoropy(z_out[d-1], yi))
        ##### 逆伝播
        dd=np.zeros((d+1,q))
        for j in reversed(range(0,d)):
            dd[j]=backward(w_hidden,dd[j+1],w_out,z_out[j]-yi,U[j])
        dd=np.delete(dd,d,0)
        ##### パラメータの更新
        xi2=np.insert(xi,0,1,axis=1)
        z2=np.insert(Z,0,1,axis=1)
        zti=np.delete(Z,d,0)
        zlast=np.reshape(z2[d,],(z2.shape[1],1))
        zy=np.reshape(z_out[d-1]-yi,(len(yi),1))
        z2ti=np.delete(z2,d,0)
        
        w_in=w_in-eta_t*np.dot(dd.T,xi2)
        w_hidden=w_hidden-eta_t*np.dot(dd.T,zti)
        w_out=w_out-eta*np.dot(zy,zlast.T)
        
    ##### エポックごとの訓練誤差: eの平均をerrorにappendする
    error.append(sum(e)/n)
    e = []
    
    #####: 誤差をe_testにappendする
    for j in range(0,n_test):
        xi = x_test[j, :, :]
        yi = y_test[j, :]
        Z=np.zeros((d+1,q))
        U=np.zeros((d,q))
        z_out=np.zeros((d,m))
        for j in range(1,d+1):
            Z[j],U[j-1]=forward(np.append(1, xi[j-1]), Z[j-1], w_in, w_hidden, sigmoid)
            z_out[j-1] = softmax(np.dot(w_out, np.append(1, Z[j])))
            
        e_test.append(CrossEntoropy(z_out[d-1], yi)) 
    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする
    error_test.append(sum(e_test)/n_test)
    e_test = []

########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight', transparent=True)

########## 確率が高いクラスにデータを分類
prob = []
for j in range(0, n_test):    
    xi = x_test[j, :, :]
    yi = y_test[j, :]
    Z=np.zeros((d+1,q))
    U=np.zeros((d,q))
    z_out=np.zeros((d,m))
    ##### 順伝播
    for j in range(d+1):
        Z[j], U[j-1] = forward(np.append(1, xi[j-1]), Z[j-1], w_in, w_hidden, sigmoid)
        z_out[j-1] = softmax(np.dot(w_out, np.append(1, Z[j])))
    
    prob.append(z_out[d-1])

predict = np.argmax(prob, 1)

##### confusion matrixと誤分類結果のプロット
ConfMat = np.zeros((m, m))
for i in range(m):
    idx_true = (y_test[:, i]==1)
    for j in range(m):
        idx_predict = (predict==j)
        ConfMat[i, j] = sum(idx_true*idx_predict)
        if j != i:
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = x_test[l, :, :]
                sns.heatmap(D, cbar =False, cmap="Blues", square=True)
                plt.axis("off")
                plt.title('{} to {}'.format(i, j))
                plt.savefig("./misslabeled{}.pdf".format(l), bbox_inches='tight', transparent=True)

plt.clf()
fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)
