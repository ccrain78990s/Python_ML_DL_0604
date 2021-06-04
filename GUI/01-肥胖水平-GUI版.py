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

"""
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


win = tk.Tk()
def event1():
    # Entry 輸入框的變數
    global entry1,entry2,entry3,entry4,entry5
    global entry6, entry7, entry8, entry9, entry10
    global entry11, entry12, entry13, entry14
    # Label3 的文字 變數
    global label3Str

    v1=entry1.get()     #取得用戶所輸入的文字
    v2=entry2.get()     #取得用戶所輸入的文字
    v3=entry3.get()     #取得用戶所輸入的文字
    v4=entry4.get()     #取得用戶所輸入的文字
    v5 = entry5.get()  # 取得用戶所輸入的文字
    v6 = entry6.get()  # 取得用戶所輸入的文字
    v7 = entry7.get()  # 取得用戶所輸入的文字
    v8 = entry8.get()  # 取得用戶所輸入的文字
    v9 = entry9.get()  # 取得用戶所輸入的文字
    v10 = entry10.get()  # 取得用戶所輸入的文字
    v11= entry11.get()  # 取得用戶所輸入的文字
    v12= entry12.get()  # 取得用戶所輸入的文字
    v13= entry13.get()  # 取得用戶所輸入的文字
    v14= entry14.get()  # 取得用戶所輸入的文字

    #------- tensorflow MLP
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
    x_test = np.array([[float(v1),float(v2),float(v3),float(v4),
                        float(v5),float(v4),float(v4),float(v4),
                        float(v9),float(v10),float(v11),float(v4),
                        float(v13),float(v14)]])

    predict = model.predict(x_test)
    print("預測:", predict)
    print("結果:", np.argmax(predict[0]))

    predict2 = model.predict_classes(x_test)
    print("預測分類:", predict2)
    array1=["體重不足","體重正常",
            "超重級別I","超重級別II",
            "肥胖類型I","肥胖類型II","肥胖類型II"]
    dict1={"0":"體重不足","1":"體重正常 ",
           "2":"超重級別I","3":"超重級別II",
           "4":"肥胖類型I","5":"肥胖類型II","6":"肥胖類型III"}
    t1=predict2[0]
    t1=dict1[str(t1)]

    label3Str.set("預設答案:"+str(predict2[0])+" , "+str(dict1[str(predict2[0])]))    #印出選單的內容

########################################################
win.wm_title("肥胖水平的預測")                 # 設定抬頭名稱
win.minsize(width=800, height=315)   # 320,200
win.resizable(width=False, height=False) # 是否可以改變視窗大小
########################################################
#背景
background_image = ImageTk.PhotoImage(Image.open("bg1.jpg"))
background_label = tk.Label(win, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

########################################################

"""
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
#輸入框1
t=-70
label1=tk.Label(win,text="性別:")
label1.place(x=10,y=80+t)
entry1=tk.Entry(win)
entry1.place(x=10,y=80+t+25)
#輸入框2
t=-30
label2=tk.Label(win,text="年齡:")
label2.place(x=10,y=80+t)
entry2=tk.Entry(win)
entry2.place(x=10,y=80+t+25)
#輸入框3
t=10
label3=tk.Label(win,text="家庭成員是否患有或患有超重")
label3.place(x=10,y=80+t)
entry3=tk.Entry(win)
entry3.place(x=10,y=80+t+25)
#輸入框4
t=50
label4=tk.Label(win,text="經常吃高熱量的食物嗎")
label4.place(x=10,y=80+t)
entry4=tk.Entry(win)
entry4.place(x=10,y=80+t+25)
#輸入框5
t=90
label5=tk.Label(win,text="您平時吃飯時吃蔬菜嗎")
label5.place(x=10,y=80+t)
entry5=tk.Entry(win)
entry5.place(x=10,y=80+t+25)
#輸入框6
t=-70
label6=tk.Label(win,text="你每天吃幾頓主食")
label6.place(x=200,y=80+t)
entry6=tk.Entry(win)
entry6.place(x=200,y=80+t+25)
#輸入框7
t=-30
label7=tk.Label(win,text="您在兩餐之間吃任何食物嗎")
label7.place(x=200,y=80+t)
entry7=tk.Entry(win)
entry7.place(x=200,y=80+t+25)
#輸入框8
t=10
label8=tk.Label(win,text="你抽煙嗎")
label8.place(x=200,y=80+t)
entry8=tk.Entry(win)
entry8.place(x=200,y=80+t+25)
#輸入框9
t=50
label9=tk.Label(win,text="你每天喝多少水")
label9.place(x=200,y=80+t)
entry9=tk.Entry(win)
entry9.place(x=200,y=80+t+25)
#輸入框10
t=90
label10=tk.Label(win,text="您是否監控您每天攝入的卡路里")
label10.place(x=200,y=80+t)
entry10=tk.Entry(win)
entry10.place(x=200,y=80+t+25)
#輸入框11
t=-70
label11=tk.Label(win,text="您多久進行一次體育鍛煉？")
label11.place(x=390,y=80+t)
entry11=tk.Entry(win)
entry11.place(x=390,y=80+t+25)
#輸入框12
t=-30
label12=tk.Label(win,text="您使用手機、電子遊戲、電視、電腦等技術設備的時間有多少？")
label12.place(x=390,y=80+t)
entry12=tk.Entry(win)
entry12.place(x=390,y=80+t+25)
#輸入框13
t=10
label13=tk.Label(win,text="你多久喝一次酒？")
label13.place(x=390,y=80+t)
entry13=tk.Entry(win)
entry13.place(x=390,y=80+t+25)
#輸入框14
t=50
label14=tk.Label(win,text="您通常使用哪種交通工具？")
label14.place(x=390,y=80+t)
entry14=tk.Entry(win)
entry14.place(x=390,y=80+t+25)
########################################################
#按鈕
btn1 =tk.Button(win,text="預測",command=event1,font=20)
btn1.place(x=30,y=220)
########################################################
#印出預測結果
label3Str=StringVar()
label3=tk.Label(win,text="答案",textvariable=label3Str)
label3.place(x=30,y=250)         #印出選單的內容

win.mainloop()


