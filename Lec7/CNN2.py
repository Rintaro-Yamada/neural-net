#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Activation, Flatten, Dense
from keras import optimizers
from keras.optimizers import Adam

##### データの取得
#クラス数を定義
m = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train[y_train < m, :, :, np.newaxis]

x_test = x_test.astype('float32') / 255.
x_test = x_test[y_test < m, :, :, np.newaxis]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d, _, _ = x_train.shape
n_test, _, _, _ = x_test.shape

########## モデルの定義: model = Model() または model = Sequential()を用いる
model = Sequential()
model.add((Conv2D(9,(3,3),padding="same",activation="relu",input_shape=(28,28,1))))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(m, activation="softmax"))


########## モデルのプロット
#plot_model(model, to_file = "./model.png", show_shapes=True)

########## オプティマイザの定義
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

########## パラメータ推定: history = model.fit()で推定結果をhistoryに保存する
history = model.fit(x_train,y_train,batch_size=16, nb_epoch=10, verbose=1,validation_split=0.1)

########## 結果のプロット
##### 誤差関数の推移
plt.clf()
plt.plot(history.history['loss'], label="training", lw=3)
plt.plot(history.history['val_loss'], label="test", lw=3)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight')

##### 分類精度
plt.clf()
plt.plot(history.history['acc'], label="training", lw=3)
plt.plot(history.history['val_acc'], label="test", lw=3)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./accuracy.pdf", bbox_inches='tight', transparent=True)

##### confusion matrixと誤分類結果のプロット
predict = np.argmax(model.predict(x_test), 1)
ConfMat = np.zeros((m, m))
for i in range(m):
    idx_true = (y_test[:, i]==1)
    for j in range(m):
        idx_predict = (predict==j)
        ConfMat[i, j] = sum(idx_true*idx_predict)
        if j != i:
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = x_test[l, :, :, 0]
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