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

#read data
df = pd.read_excel ('桃園全區實價登錄資料_20190430修改後.xls',0)

print(df.head())
print(df.columns)
print(df.index)
print(df.columns)

#去除空白
# df['DataFrame Column'] = df['DataFrame Column'].fillna(0)
#全部的空白
df=df.fillna(-1)

columnsName=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
columnsName=['鄉鎮市區','交易標的','土地區段位置建物區段門牌',
             # '土地移轉總面積平方公尺',
             '都市土地使用分區','非都市土地使用分區','非都市土地使用編定',
             #'交易年月日','年','月',
            '交易筆棟數','移轉層次','總樓層數','建物型態','主要用途','主要建材',
             # '建築完成年月','建物移轉總面積平方公尺','建物現況格局-房','建物現況格局-廳','建物現況格局-衛',
            '建物現況格局-隔間','有無管理組織',
             #'總價元','單價元平方公尺',
            '車位類別',
             #'車位移轉總面積平方公尺','車位總價元',
            '備註','編號']

for x in columnsName:
    df[x+'_Code'] = df[x].astype("category").cat.codes


print(df.head())

df.to_excel("桃園全區實價登錄資料_20190430修改後_轉數字.xls")


