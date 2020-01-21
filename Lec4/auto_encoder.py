#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist

##### データの取得
#クラス数
m = 1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape([60000, 28*28])
x_train = x_train[y_train < m,:]

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape([10000, 28*28])
x_test = x_test[y_test < m,:]

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)

##### 
def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp, tmp*(1-tmp)

def ErrorFunction(x, y):
    # returnの後に二乗誤差関数を返すプログラムを書く
    return (y-x).T*(y-x)

def forward(x, param, fncs):
    tmp = np.dot(param, x)
    z = np.append(1, fncs(tmp)[0])
    u = fncs(tmp)[1]
    return z, u

def backward(param, delta, derivative):
    return np.dot(param.T, delta)*derivative

def adam(param, m, v, error, t, 
         alpha = 0.001, beta1 = 0.9, beta2 = 0.999, tol = 10**(-8)):
    # returnの後にadamによるパラメータ更新のプログラムを書く
    m=beta1*m+(1-beta1)*error
    v=beta2*v+(1-beta2)*error*error
    m1=m/(1-beta1**t)
    v1=v/(1-beta2**t)
    param=param-alpha*m1/(np.sqrt(v1)+tol)
    return param,m,v

#####中間層のユニット数とパラメータの初期値
N1 = 64

w0 = np.random.normal(0, 0.3, size=(N1, d+1))
w1 = np.random.normal(0, 0.3, size=(d, N1+1))

########## 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

num_epoch = 10

##### adamの初期値
m0 = np.zeros(shape=(N1, d+1))
v0 = np.zeros(shape=(N1, d+1))
m1 = np.zeros(shape=(d, N1+1))
v1 = np.zeros(shape=(d, N1+1))

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    print(epoch)
    for i in index:
        
        ##### データに加えるノイズ
        noise =np.random.normal(0,1,len(x_train[i, :]))
        xi = np.append(1, x_train[i, :] + noise)
        
        
        ##### 順伝播
        z1, u1 = forward(xi, w0, sigmoid)
        z2 = np.dot(w1, z1)
               
        ##### 誤差評価
        e.append(ErrorFunction(z2, x_train[i, :]))
        
        ##### 逆伝播
        delta2 = z2 - x_train[i, :]
        delta1 = backward(w1[:, 1:], delta2, u1)
        
        ##### adamによるパラメータの更新
        #print(xi.shape)
        #print(delta1.shape)
        #print(z2.shape)
        #print(delta2.shape)
        
        xi=np.reshape(xi,(len(xi),1))
        delta1=np.reshape(delta1,(len(delta1),1))
        z1=np.reshape(z1,(len(z1),1))
        delta2=np.reshape(delta2,(len(delta2),1))
        
        w0,m0,v0=adam(w0,m0,v0,np.dot(delta1,xi.T),epoch+1)
        w1,m1,v1=adam(w1,m1,v1,np.dot(delta2,z1.T),epoch+1)
    
    ##### training error
    error.append(sum(e)/n)
    e = []
    
    ##### test error
    for j in range(0, n_test):
        ##### データに加えるノイズ
        noise =np.random.normal(0,1,len(x_test[j, :]))            
        xi = np.append(1, x_test[j, :] + noise)
        
        z1, u1 = forward(xi, w0, sigmoid)
        z2 = np.dot(w1, z1)
        
        e_test.append(ErrorFunction(z2, x_test[j, :]))
    
    error_test.append(sum(e_test)/n_test)
    e_test = []


########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", transparent=True)

##### 0からn-1までの自然数のうち, 好きな数をjに代入する
j =1

xi = np.append(1, x_test[j, :] + np.random.normal(0, 0.05, size= d))
z1, u1 = forward(xi, w0, sigmoid)
z2 = np.dot(w1, z1)

D_o = np.reshape(x_test[j, :], (28, 28))
D = np.reshape(z2, (28, 28))


plt.clf()
sns.heatmap(D_o, cbar =False, cmap="Blues", square=True)
plt.axis("off")
plt.savefig("./original.pdf", bbox_inches='tight', transparent=True)

plt.clf()
sns.heatmap(D, cbar =False, cmap="Blues", square=True)
plt.axis("off")
plt.savefig("./reconstruct.pdf", bbox_inches='tight', transparent=True)

plt.clf()
fig,axes = plt.subplots(nrows=8,ncols=8,figsize=(10, 10))
for i in range(8):
    for j in range(8):
        D = np.reshape(w0[i, 1:], (28, 28)).T
        axes[i,j].imshow(D)
        axes[i,j].axis("off")
plt.savefig("./layer1.pdf", bbox_inches='tight', transparent=True)

plt.clf()
fig,axes = plt.subplots(nrows=8,ncols=8,figsize=(10, 10))
for i in range(8):
    for j in range(8):
        D = np.reshape(w1[:, i], (28, 28)).T
        axes[i,j].imshow(D)
        axes[i,j].axis("off")
plt.savefig("./layer2.pdf", bbox_inches='tight', transparent=True)


