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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
##############################################################################
# 資料讀取、匯入、整理
import pandas as pd
#read data
df = pd.read_excel('obesity_levels.xls',0)
print("##################試印資料前幾筆##################")
print(df.head())
print("##################資料欄位名稱##################")
print(df.columns)
print("##################資料筆數##################")
print(df.index)
print("##################df相關資訊##################")
print(df.info)
print("##################統計描述##################")
print(df.describe())

# 註:文字分類 轉 數字分類 已在另一檔案轉好存入新excel表中
columnsName=['Gender_Code', 'Age', 'Height', 'Weight',
              'family_history_with_overweight_Code','FAVC_Code',
              'FCVC', 'NCP','CAEC_Code', 'SMOKE_Code', 'CH2O',
              'SCC_Code', 'FAF', 'TUE','CALC_Code', 'MTRANS_Code',
             'NObeyesdad_Code']
print(df.columns)
##############################################################################
# 資料拆切
import numpy as np
from sklearn.model_selection import train_test_split

print("資料拆切---")
# 決定X 分類 和Y分類 要用的欄位
# 2D 特徵值 共16種
dfX=df[['Gender_Code', 'Age', 'Height', 'Weight',
        'family_history_with_overweight_Code','FAVC_Code',
        'FCVC', 'NCP','CAEC_Code', 'SMOKE_Code', 'CH2O',
        'SCC_Code', 'FAF', 'TUE','CALC_Code', 'MTRANS_Code']]
# 1D 標籤 肥胖水平
dfY=df['NObeyesdad_Code']

print("資料前五項預覽")
print(df.head(5))

X=dfX.to_numpy()
Y=dfY.to_numpy()

# 均一化 加上資料上下變化可能很大 讓資料更漂亮  最後預測值記得轉回來
dfX = (dfX - dfX.min()) / (dfX.max() - dfX.min())

t1=Y.shape[0]
Y=np.reshape(Y,(t1,))  # 2D 轉 1D

X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size=0.1)
print("資料拆切---OK")
##############################################################################
