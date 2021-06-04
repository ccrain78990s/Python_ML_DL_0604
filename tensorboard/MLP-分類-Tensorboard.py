#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "Chen"

"""
資料來源：
Estimation of obesity levels based on eating habits and physical condition Data Set
(根據來自哥倫比亞、秘魯和墨西哥的個人的飲食習慣和身體狀況估計肥胖水平的數據集)
https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+
簡單數據說明:
16特徵值，1label(7類別)
與飲食習慣有關的屬性包括：
經常食用高熱量食物（FAVC），蔬菜食用頻率（FCVC），主餐次數（NCP），
兩餐之間的食物消耗量（CAEC），每日水消耗量（CH20） ) 和酒精消耗量 (CALC)。
與身體狀況相關的屬性是：
卡路里消耗監測（SCC）、身體活動頻率（FAF）、使用技術設備的時間（TUE）、
使用的交通工具（MTRANS）
其他變量是：
性別、年齡、身高和體重。
最後，所有數據都被標記並創建了類變量 NObesity，其值是：
體重不足、正常體重、超重級別 I、超重級別 II、肥胖類型 I、肥胖類型 II 和肥胖類型 III
"""
"""
tensorboard運行步驟
開啟cmd > 移動路徑至此py檔位置 > 打上指令tensorboard --logdir=logs/ > 貼上結果網址
"""

# 去掉警告紅字
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
import xlrd
import xlwt
import numpy as np
import pandas as pd
#
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
#
import seaborn as sns
import matplotlib.pyplot as plt
#
import tensorflow as tf
from tensorflow.keras.models import model_from_json
#
from tensorflow.keras.callbacks import TensorBoard  #######

#read data
df = pd.read_excel('obesity_levels.xls',0)



#############
print("資料拆切---")
# 決定X 分類 和Y分類 要用的欄位
# 2D
"""
dfX=df[['Gender_Code', 'Age', 'Height', 'Weight',
        'family_history_with_overweight_Code','FAVC_Code',
        'FCVC', 'NCP','CAEC_Code', 'SMOKE_Code', 'CH2O',
        'SCC_Code', 'FAF', 'TUE','CALC_Code', 'MTRANS_Code']]
"""
dfX=df[['Gender_Code', 'Age',
        'family_history_with_overweight_Code','FAVC_Code',
        'FCVC', 'NCP','CAEC_Code', 'SMOKE_Code', 'CH2O',
        'SCC_Code', 'FAF', 'TUE','CALC_Code', 'MTRANS_Code']]
# 1D
dfY=df['NObeyesdad_Code']

# 均一化 讓資料更漂亮  最後預測值記得轉回來
dfX = (dfX - dfX.min()) / (dfX.max() - dfX.min())

X=dfX.to_numpy()
Y=dfY.to_numpy()
X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.1)


x_train = X_train[:, :]
x_test =X_test[:, :]

y_train  = Y_train
y_test =Y_test


dim=14
category=7
t=2
#One-hot encoding
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))


# 讀取模型架構
try:
    # 如果有類神經權重檔案的時候，就直接打開讀取使用
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # 讀取模型權重
        model.load_weights("model.h5")
        # 註 : 讀取完權重記得下面也要寫
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'])
except IOError:
    # 相反地，若是沒有的話，就重新訓練。
    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=200,
        activation=tf.nn.relu,
        input_dim=dim))
    model.add(tf.keras.layers.Dense(units=80*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=100*t,
        activation=tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units=category,
        activation=tf.nn.softmax ))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs")   #######
    model.fit(x_train, y_train2,
              epochs=300*t,
              batch_size=64,
              callbacks=[tensorboard],          #######
              verbose=1)

# 模型摘要
model.summary()
# 測試
score = model.evaluate(x_test, y_test2, batch_size=64)
print("score:",score)
# 預測結果
predict2 = model.predict_classes(x_test)
print("預測的分類:",predict2)
print("實際的分類",y_test[:])



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



