#!/usr/bin/env python
# -*- coding=utf-8 -*-
# pip install seaborn
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

"""
 資料來源：
 https://moptt.tw/p/home-sale.M.1557147662.A.2A6
 https://lvr.land.moi.gov.tw/
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
#df = pd.read_excel('桃園全區實價登錄資料_20190430修改後_轉數字.xls',0)
df = pd.read_excel('桃園全區實價登錄資料_20190430修改後_轉數字_lite.xls',0)

#全部的空白
df=df.fillna(-1)

print(df.head())
print(df.columns)
print(df.index)
print(df.columns)

#columnsName=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
#columnsName=['B','D','F','G','H','I','J','N','O']
#columnsName=['A','E','K','L','M','B_Code','D_Code','F_Code','G_Code','H_Code','I_Code','J_Code','N_Code']

columnsName=['鄉鎮市區_Code','交易標的_Code','土地區段位置建物區段門牌_Code',
             '土地移轉總面積平方公尺',
             '都市土地使用分區_Code','非都市土地使用分區_Code','非都市土地使用編定_Code',
             '交易年月日','年','月',
            '交易筆棟數_Code','移轉層次_Code','總樓層數_Code','建物型態_Code','主要用途_Code','主要建材_Code',
            '建築完成年月','建物移轉總面積平方公尺','建物現況格局-房','建物現況格局-廳','建物現況格局-衛',
            '建物現況格局-隔間_Code','有無管理組織_Code',
            # '總價元','單價元平方公尺',
            '車位類別_Code',
            '車位移轉總面積平方公尺','車位總價元',
            '備註_Code','編號_Code']


#############
print("資料拆切---")
# 決定X 分類 和Y分類 要用的欄位
#dfX=df[['A','E','K','L','M','B_Code','D_Code','F_Code','G_Code','H_Code','I_Code','J_Code','N_Code']]
dfX=df[columnsName]
dfY=df["總價元"]
dfY2=dfY


dfX = (dfX - dfX.min()) / (dfX.max() - dfX.min())  # 均一化
dfY = (dfY - dfY2.min()) / (dfY2.max() - dfY2.min())  # 均一化


X=dfX.to_numpy()
Y=dfY.to_numpy()
X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.01)




x_train = X_train[:, :]
x_test =X_test[:, :]

y_train  = Y_train
y_test =Y_test


dim=x_train.shape[1]
category=1
t=1
#y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
#y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:4]",x_train[:4])
print("y_train[:4]",y_train[:4])
# print("y_train2[:4]",y_train2[:4])




# 讀取模型架構
try:
    with open('model2.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # 讀取模型權重
        model.load_weights("model2.h5")
        learning_rate = 0.0001
        opt1 = tf.keras.optimizers.Nadam(lr=learning_rate)
        model.compile(loss='mse', optimizer=opt1, metrics=['mae'])
except IOError:

    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(320, activation='tanh', input_shape=[x_train.shape[1]]))
    model.add(tf.keras.layers.Dense(640, activation='tanh'))
    model.add(tf.keras.layers.Dense(640, activation='tanh'))
    model.add(tf.keras.layers.Dense(1))

    learning_rate = 0.0001
    opt1 = tf.keras.optimizers.Nadam(lr=learning_rate)
    model.compile(loss='mse', optimizer=opt1, metrics=['mae'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint("model2.h5", monitor='loss', verbose=1,
                                                    save_best_only=True, mode='auto', period=1)

    # 保存模型架構
    with open("model2.json", "w") as json_file:
        json_file.write(model.to_json())

    history = model.fit(x_train, y_train,
                        epochs=4000,
                        callbacks=[checkpoint],
                        batch_size=100)

    # 保存模型權重
    model.save_weights("model2.h5")

    import matplotlib.pyplot as plt

    print(history.history.keys())
    plt.plot(history.history['mae'],label="mae")
    plt.plot(history.history['loss'],label="loss")
    plt.title('house price')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.show()

#測試
model.summary()

score = model.evaluate(x_test, y_test, batch_size=64)
print("score:",score)

predict2 = model.predict(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])



#保存模型架構
with open("model.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model.h5")


 




#############
"""
print("畫seaborn 圖---")



df2=df[["O_Code",'A','E','K','L','M','B_Code','D_Code','F_Code','G_Code','H_Code','I_Code','J_Code','N_Code']]
df2=df[["O_Code",'A','E','M','B_Code']]


sns.set_theme(style="ticks")
sns.pairplot(df2, hue="O_Code")
plt.savefig("seaborn.jpg")
plt.show()
"""



