"""
參考資料：
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


"""
import pickle

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
#LinearRegression()
print(reg.coef_)

# array([0.5, 0.5])


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

year = np.arange(0, 100, 1)
year = np.reshape(year, (1, -1))
year_predict = np.arange(100, 200, 1)
year_predict = np.reshape(year_predict, (1, -1))

y = np.sin(2 * np.pi * year / 15) + np.cos(2 * np.pi * year / 15)
"""
lm = LinearRegression()
lm.fit(year, y)
"""


# load the model from disk

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
result = model.score(year, y)
print(result)




y_pred = model.predict(year_predict)

plt.plot(year[0, :], y[0, :])
plt.plot(year_predict[0, :], y_pred[0, :])
plt.ylabel('np.sin(2*pi*year/15)+np.cos(2*pi*year/15)')
plt.show()
