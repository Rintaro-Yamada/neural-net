#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第6回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

#####
def ReLU(x):
    return x*(x>0)

def Convolution(img, param, bias, fncs, channel, padding, stride):
    ##### 課題1-(a). 畳み込み層の計算を完成させる
    z = np.pad(img,(padding,padding), 'constant')   #イメージのパディング
    H=param.shape[2]    #フィルタのサイズ
    W=z.shape[0]    #imgのパディング後のサイズ
    z=z[:,:,np.newaxis]  #次元増やす  
    I=(W-H)//stride+1
    J=(W-H)//stride+1
    Z=np.zeros((I,I,channel))
    u_out=0
    
    for k in range(0,channel):
        for i in range(0,I):
            for j in range(0,J):           
                for p in range(0,H):
                    for q in range(0,H):
                        u_out+=param[p,q,k]*z[stride*i+p,stride*j+q,:]
                        
                Z[i,j,k]=u_out
                u_out=0.0            
    return Z

def MaxPooling(img, filter_size, channel, stride):
    ##### 課題1-(b). maxプーリングの計算を完成させる
    I=(img.shape[1]-filter_size)//stride+1
    Z=np.zeros((I,I,channel))
    m=0.0
    for k in range(0,channel):
        for i in range(0,I):
            for j in range(0,I):
                for p in range(filter_size):
                    for q in range(filter_size):
                        if m<img[stride*i+p,stride*j+q,k]:
                            m=img[stride*i+p,stride*j+q,k]
                Z[i,j,k]=m
                m=0.0           
    return Z


##### データの読み込み
img = np.load("./img.npy")

##### 課題2-(a)
w = np.array([[[-1,-1,0],[-1,0,1],[0,1,1]],[[0,1,0],[1,-4,1],[0,1,0]],[[1,1,1],[1,1,1],[1,1,1]]])
print(w)

##### 課題2-(b)
##### 畳み込み
C = Convolution(img,w,0,ReLU,1,2,1)

##### プーリング
P = MaxPooling(C,3,3,3)

########## 結果の出力
plt.clf()
sns.heatmap(img, cbar =False, cmap="CMRmap", square=True)
plt.axis("off")
plt.savefig("./original.pdf", bbox_inches='tight', transparent = True)

for i in range(0,C.shape[2]):
    plt.clf()
    sns.heatmap(C[:,:,i], cbar =False, cmap="CMRmap", square=True)
    plt.axis("off")
    plt.title('Convolution1: axis{}'.format(i))
    plt.savefig("./conv1-{}.pdf".format(i), bbox_inches='tight', transparent = True)

for i in range(0,P.shape[2]):
    plt.clf()
    sns.heatmap(P[:,:,i], cbar =False, cmap="CMRmap", square=True)
    plt.axis("off")
    plt.title('Pooling: axis{}'.format(i))
    plt.savefig("./pooling{}.pdf".format(i), bbox_inches='tight', transparent = True)    





