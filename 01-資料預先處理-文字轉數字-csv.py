#!/usr/bin/env python
# -*- coding=utf-8 -*-
# pip install seaborn
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

# 讀取資料
df = pd.read_csv('obesity_levels.csv')

print(df.head())
print(df.columns)
print(df.index)


#columnsName=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
columnsName=['Gender', #'Age', 'Height', 'Weight',
              'family_history_with_overweight','FAVC',
              #'FCVC', 'NCP',
              'CAEC', 'SMOKE', #'CH2O',
              'SCC', #'FAF', 'TUE',
              'CALC', 'MTRANS', 'NObeyesdad']

for x in columnsName:
    df[x+'_Code'] = df[x].astype("category").cat.codes

#df['Spectral_Class_Code'] = df['Spectral_Class'].astype("category").cat.codes

print(df.head())

df.to_excel("obesity_levels.xls")


