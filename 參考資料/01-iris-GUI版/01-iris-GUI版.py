#!/usr/bin/python
import tkinter as tk
from tkinter import ttk
from tkinter import StringVar
from PIL import ImageTk, Image

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from time import time
from tensorflow.keras.models import model_from_json


win = tk.Tk()
def event1():
    global entry1       # Entry 輸入框的變數
    global entry2       # Entry 輸入框的變數
    global entry3       # Entry 輸入框的變數
    global entry4       # Entry 輸入框的變數
    global label3Str    # Label3 的文字 變數

    v1=entry1.get()     #取得用戶所輸入的文字
    v2=entry2.get()     #取得用戶所輸入的文字
    v3=entry3.get()     #取得用戶所輸入的文字
    v4=entry4.get()     #取得用戶所輸入的文字

    #------- tensorflow

    # 讀取模型架構
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # 讀取模型權重
    model.load_weights("model.h5")

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    # 測試
    # x_test = np.array([[5.8, 2.8, 5.1, 2.4], [7.2, 3.2, 6.0, 1.8]])
    x_test = np.array([[float(v1),float(v2),float(v3),float(v4)]])

    predict = model.predict(x_test)
    print("predict:", predict)
    print("Ans:", np.argmax(predict[0]))

    predict2 = model.predict_classes(x_test)
    print("predict_classes:", predict2)
    array1=["Iris Setosa","ris Versicolour ","Iris Virginica"]
    dict1={"0":"Iris Setosa","1":"ris Versicolour ","2":"Iris Virginica"}
    t1=predict2[0]
    t1=dict1[str(t1)]

    label3Str.set("預設答案:"+str(predict2[0])+" , "+str(dict1[str(predict2[0])]))    #印出選單的內容


win.wm_title("iris花的預測")                 # 設定抬頭名稱
win.minsize(width=627, height=315)   # 320,200
win.resizable(width=False, height=False) # 是否可以改變視窗大小

background_image = ImageTk.PhotoImage(Image.open("background.png"))
background_label = tk.Label(win, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

t=-70
label1=tk.Label(win,text="花瓣寬度:")
label1.place(x=10,y=80+t)
entry1=tk.Entry(win)
entry1.place(x=70,y=80+t)

t=-30
label2=tk.Label(win,text="花瓣高度:")
label2.place(x=10,y=80+t)
entry2=tk.Entry(win)
entry2.place(x=70,y=80+t)


t=10
label3=tk.Label(win,text="花萼寬度:")
label3.place(x=10,y=80+t)
entry3=tk.Entry(win)
entry3.place(x=70,y=80+t)

t=50
label4=tk.Label(win,text="花萼高度:")
label4.place(x=10,y=80+t)
entry4=tk.Entry(win)
entry4.place(x=70,y=80+t)



btn1 =tk.Button(win,text="預測",command=event1)
btn1.place(x=30,y=250)



label3Str=StringVar()
label3=tk.Label(win,text="答案",textvariable=label3Str)
label3.place(x=30,y=220)         #印出選單的內容

win.mainloop()


