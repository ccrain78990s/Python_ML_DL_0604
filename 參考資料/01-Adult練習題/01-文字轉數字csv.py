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
df = pd.read_csv('adult.csv')

print(df.head())
print(df.columns)
print(df.index)
print(df.columns)

columnsName=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
columnsName=['B','D','F','G','H','I','J','N','O']

for x in columnsName:
    df[x+'_Code'] = df[x].astype("category").cat.codes

#df['Spectral_Class_Code'] = df['Spectral_Class'].astype("category").cat.codes

print(df.head())

df.to_excel("adult.xlsx")


