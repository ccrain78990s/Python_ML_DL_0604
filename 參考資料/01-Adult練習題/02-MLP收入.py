#!/usr/bin/env python
# -*- coding=utf-8 -*-
# pip install seaborn
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

"""
 資料來源：
 https://archive.ics.uci.edu/ml/datasets/Adult
"""
import xlrd
import xlwt
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import model_from_json

#read data
df = pd.read_excel('adult.xlsx',0)

print(df.head())
print(df.columns)
print(df.index)
print(df.columns)

#columnsName=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
#columnsName=['B','D','F','G','H','I','J','N','O']
columnsName=['A','E','K','L','M','B_Code','D_Code','F_Code','G_Code','H_Code','I_Code','J_Code','N_Code']


#############
print("資料拆切---")
# 決定X 分類 和Y分類 要用的欄位
dfX=df[['A','E','K','L','M','B_Code','D_Code','F_Code','G_Code','H_Code','I_Code','J_Code','N_Code']]
dfY=df["O_Code"]



dfX = (dfX - dfX.min()) / (dfX.max() - dfX.min())  # 均一化


X=dfX.to_numpy()
Y=dfY.to_numpy()
X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.1)




x_train = X_train[:, :]
x_test =X_test[:, :]

y_train  = Y_train
y_test =Y_test


dim=x_train.shape[1]
category=2
t=1
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:4]",x_train[:4])
print("y_train[:4]",y_train[:4])
print("y_train2[:4]",y_train2[:4])



# 讀取模型架構
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # 讀取模型權重
        model.load_weights("model.h5")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'])
except IOError:

    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=200,
        activation=tf.nn.relu,
        input_dim=dim))
    model.add(tf.keras.layers.Dense(units=40*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=80*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=100*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=category,
        activation=tf.nn.softmax ))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    model.fit(x_train, y_train2,
              epochs=300*t,
              batch_size=64)

#測試
model.summary()

score = model.evaluate(x_test, y_test2, batch_size=64)
print("score:",score)

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])



#保存模型架構
with open("model.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model.h5")


 




#############
print("畫seaborn 圖---")



df2=df[["O_Code",'A','E','K','L','M','B_Code','D_Code','F_Code','G_Code','H_Code','I_Code','J_Code','N_Code']]
df2=df[["O_Code",'A','E','M','B_Code']]


sns.set_theme(style="ticks")
sns.pairplot(df2, hue="O_Code")
plt.savefig("seaborn.jpg")
plt.show()




