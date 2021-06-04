#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Powen Ko, www.powenko.com"

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from time import time
from tensorflow.keras.models import model_from_json

# 讀取模型架構
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# 讀取模型權重
model.load_weights("model.h5")

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])


#測試
x_test=np.array([[5.8,2.8,5.1,2.4], [7.2,3.2,6.0,1.8]])

predict = model.predict(x_test)
print("predict:",predict)
print("Ans:",np.argmax(predict[0]))

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)


