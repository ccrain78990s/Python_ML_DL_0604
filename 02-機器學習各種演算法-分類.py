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
df = pd.read_excel('obesity_levels.xls',0)
"""
print("試印資料前幾筆")
print(df.head())
print("資料欄位名稱")
print(df.columns)
print("資料筆數")
print(df.index)
"""
"""
# 註:文字分類 轉 數字分類 已在另一檔案轉好存入新excel表中
columnsName=['Gender_Code', 'Age', 'Height', 'Weight',
              'family_history_with_overweight_Code','FAVC_Code',
              'FCVC', 'NCP','CAEC_Code', 'SMOKE_Code', 'CH2O',
              'SCC_Code', 'FAF', 'TUE','CALC_Code', 'MTRANS_Code',
             'NObeyesdad_Code']
"""
##############################################################################
# 資料拆切
import numpy as np
from sklearn.model_selection import train_test_split

print("資料拆切---")
# 決定X 分類 和Y分類 要用的欄位
# 2D 特徵值 共14種(扣掉身高、體重)
dfX=df[['Gender_Code', 'Age', #'Height', 'Weight',
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
"""
import matplotlib.pyplot as plt
# seabron
# 畫seaborn圖
df111 = df[['Gender_Code', 'Age', 'Height', 'Weight',
        'family_history_with_overweight_Code','FAVC_Code',
        'FCVC', 'NCP','CAEC_Code', 'SMOKE_Code', 'CH2O',
        'SCC_Code', 'FAF', 'TUE','CALC_Code', 'MTRANS_Code','NObeyesdad_Code']]
df222=df111[0:]
import seaborn as sns
sns.set_theme(style="ticks")
sns.pairplot(df222,hue='NObeyesdad_Code')
plt.savefig('seaborn.png')
plt.show()
"""
"""
import matplotlib.pyplot as plt
# seabron
# 畫seaborn圖
df111 = df[['Gender_Code', 'Age', 'Height', 'Weight',
         'CH2O', 'FAF','NObeyesdad_Code']]
df222=df111[0:]
import seaborn as sns
#sns.set_theme(style="ticks")
sns.pairplot(df222,hue='NObeyesdad_Code', kind="kde")
plt.savefig('seaborn2.png')
plt.show()
"""
"""
import matplotlib.pyplot as plt
# seabron
# 畫seaborn圖
df111 = df[['Gender_Code', 'Age',
         'CH2O', 'FAF','NObeyesdad_Code']]
df222=df111[0:]
import seaborn as sns
#sns.set_theme(style="ticks")
sns.pairplot(df222,hue='NObeyesdad_Code', kind="kde")
plt.savefig('seaborn3.png')
plt.show()
"""
# 資料量大，畫一次先儲存下來方便以後查看

##############################################################################
# 各種機器演算法

# KNN
print("**********===KNN===**********")
from sklearn.neighbors import KNeighborsClassifier
#
knn = KNeighborsClassifier(7)
knn.fit(X_train, y_train)
print('KNN準確率: %.2f' % knn.score(X_test, y_test))

#print("預測值",knn.predict(X_test))
#print("實際",y_test)

# K-means
print("**********===K-means===**********")
from sklearn.cluster import KMeans
from sklearn import metrics
#
kmeans  = KMeans(n_clusters = 7)
kmeans.fit(X_train)
y_predict=kmeans.predict(X_train)
score = metrics.accuracy_score(y_test,kmeans.predict(X_test))
print('K-means準確率:{0:f}'.format(score))

#print("預測       ",kmeans.predict(X_test))
#print("實際       ",y_test)

print("**********===決策樹===**********")
from sklearn import tree
#
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
print('決策樹準確率: %.2f' % clf.score(X_test, y_test))

# tree.export_graphviz(clf,out_file='tree.dot')
#print("預測答案:",clf.predict(X_test))
#print("實際答案:",y_test)
"""
#決策樹圖表
import matplotlib.pyplot as plt
tree.plot_tree(clf)
plt.show()
"""

# Regression 不合適做分類資料 略過

print("**********===隨機森林===**********")
from sklearn.ensemble import RandomForestClassifier
#
RForest = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=2)
RForest.fit(X_train, y_train)
print('隨機森林準確率: %.2f' % RForest.score(X_test, y_test))

# tree.export_graphviz(clf,out_file='tree.dot')
#print("預測答案:",RForest.predict(X_test))
#print("實際答案:",y_test)

print("**********===貝氏分類器===**********")
from sklearn.naive_bayes import GaussianNB
#
model = GaussianNB()
model.fit(X_train, y_train)
predicted= model.predict(X_test)
model.score(X_test,y_test)
print("貝氏分類器準確率",model.score(X_test,y_test))

#print(predicted)
#print(model.predict_proba(X_test))
#print("Number of mislabeled points out of a total %d points : %d"
#      % (X_test.shape[0], (y_test != predicted).sum()))
#print(model.class_prior_ )
#print(model.get_params() )

print("**********===SVM--SVC==**********")
from sklearn import svm
#
regr = svm.SVC()
regr.fit(X_train, y_train)
print('SVM--SVC準確率: %.2f' % regr.score(X_test, y_test))

#print("預測答案:",regr.predict(X_test))
#print("實際答案:",y_test)

print("**********===SVM--Non-linearSVC==**********")
from sklearn import svm
#
clf = svm.NuSVC(gamma='auto')
clf.fit(X_train, y_train)
print('SVM--Non-linearSVC準確率: %.2f' % clf.score(X_test, y_test))

print("**********===SVM--linearSVC==**********")
from sklearn import svm
#
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
print('SVM--linearSVC準確率: %.2f' % lin_clf.score(X_test, y_test))

print("**********===SGD==**********")
from sklearn.linear_model import SGDClassifier
#
clf = SGDClassifier(loss="log", penalty="l1", max_iter=5)
clf.fit(X_train, y_train)
print('SGD準確率: %.2f' % clf.score(X_test, y_test))

#print("預測答案:",clf.predict(X_test))
#print("實際答案:",y_test)
"""
#PCA 試寫 維度縮減
print("**********===PCA==**********")
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(X)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

"""
