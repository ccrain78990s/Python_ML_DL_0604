#!/usr/bin/python
import tkinter as tk
from tkinter import ttk
from tkinter import StringVar
from PIL import ImageTk, Image

win = tk.Tk()
def event1():
    global entry1       # Entry 輸入框的變數
    global label2Str    # Label1 的文字 變數
    global comboExample
    print(comboExample.current(), comboExample.get())
    t1=entry1.get()     #取得用戶所輸入的文字
    twd=float(t1)
    comStr=comboExample.get()
    if comStr=="美金":
        dollar=twd/28.8
    elif  comStr=="日幣":
        dollar=twd/0.26
    elif  comStr=="韓元":
        dollar=twd/0.025
    elif  comStr=="港幣":
        dollar=twd/3.64
    label2Str.set(str(dollar))
    label3Str.set(comboExample.get())   #印出選單的內容


win.wm_title("台幣轉換")                 # 設定抬頭名稱
win.minsize(width=627, height=315)   # 320,200
win.resizable(width=False, height=False) # 是否可以改變視窗大小

background_image = ImageTk.PhotoImage(Image.open("background.png"))
background_label = tk.Label(win, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


label1=tk.Label(win,text="台幣:")
label1.place(x=10,y=80)

entry1=tk.Entry(win)
entry1.place(x=50,y=80)


#buttonimage = ImageTk.PhotoImage(Image.open("buttonpic.png"))
btn1 =tk.Button(win,text="台幣轉",command=event1)
btn1.place(x=30,y=250)

label2Str=StringVar()
label2=tk.Label(win,text="美金為XXX",textvariable=label2Str)
label2.place(x=20,y=150)
label2Str.set("...")

label3Str=StringVar()
label3=tk.Label(win,text="幣別",textvariable=label3Str)
label3.place(x=160,y=150)         #印出選單的內容


comboExample = ttk.Combobox(win,
                            values=[
                                    "---選擇幣別---",
                                    "美金",
                                    "日幣",
                                    "韓元",
                                    "港幣"])
comboExample.place(x=20,y=110)
comboExample.current(0)  #選單的初始顯示

win.mainloop()



"""
#. 三聯式發票 APP
#. POS (VISA 2 TAX 5) APP
#. 



"""

