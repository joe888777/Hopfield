# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:24:33 2019

@author: asus
"""
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import glob
import math
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import glob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

dt=[]
inpdt=[]
realtest=[]
WEIGHT=[]
THETA=[]
realtrain=[]
colt=0
def sgn(u,result):
    if(result>0):
        return 1
    elif(result==0):
        return u
    else:
        return -1
def matrixmultidim(x,y):
    z=np.zeros((len(x),len(x)),dtype=np.float)
    z=list(z)
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j]=x[i]*y[j]
    return z
def setinter():
    global dt
    interface=tk.Tk()
    interface.title('Hopfield')
    interface.geometry('1100x1000')
    def selectfile():
        global dt
        dt=[]
        file=tk.filedialog.askopenfilename()
        file=open(file,'r')
        data=file.read()
       # print(data)
        data=data.split('\n')
        for i in range(len(data)):
            datatra=[]
            for j in data[i]:
                if(j==' '):
                    datatra.append(-1)
                elif(j=='1'):
                    datatra.append(1)
            dt.append(datatra)
    
        
        training(dt)
    
    def selectfileinput():
        global realtest
        global colt
        global inpdt
        inpdt=[]
        realtest=[]
        file=tk.filedialog.askopenfilename()
        file=open(file,'r')
        data=file.read()
      
        data=data.split('\n')
        for i in range(len(data)):
            fk=[]
            for j in data[i]:
                if(j==' '):
                    fk.append(-1)
                elif(j=='1'):
                    fk.append(1)
            inpdt.append(fk)
       
        #print(inpdt)
        
        realtest,colt=press2(inpdt)
    def press2(inpdt):
        
        testdata=[]
        col=0
        if(len(inpdt)>=1):
            col=len(inpdt[0])
        buf=[]
        for i in inpdt:
            if(len(i)>0):
                buf.append(i)
            else:
                testdata.append(buf)
                buf=[]
        testdata.append(buf)
        realdata=[]
            
        for i in testdata:
            onedim=[]
            for j in range(len(i)):
                for k in range(len(i[j])):
                    onedim.append(i[j][k])
            realdata.append(onedim)
        datalen=len(onedim) 
        testlabel=tk.Label(interface,text="資料筆數: "+str(len(realdata)),bg='white',font=('Arial',12),width=15,height=2)
        testlabel.grid(row=2,column=2)
        return realdata,col
    
        
    FileButton=tk.Button(interface,text="fileselect",command=selectfile)
    FileButton.grid(row=0,sticky=W)
    tstFileButton=tk.Button(interface,text="testing file select",command=selectfileinput)
    tstFileButton.grid(row=1,sticky=W)
    testlabel=tk.Label(interface,text="traindata          |        testing data          |        回想結果"+str(len(realtest)),bg='red',font=('Arial',12),width=60,height=2)
    testlabel.grid(row=2,column=1)
    testlabel=tk.Label(interface,text="資料筆數: "+str(len(realtest)),bg='white',font=('Arial',12),width=15,height=2)
    testlabel.grid(row=2,column=2)
    trainentry=tk.Entry(interface)
    trainentry.grid(row=4,column=0)
    testentry=tk.Entry(interface)
    testentry.grid(row=4,column=1)
    f =Figure(figsize=(15,10), dpi=50)
    canvas =FigureCanvasTkAgg(f, master=interface)
    #canvas.show()
    
    def training(dt):
        global WEIGHT
        global THETA
        global realtrain
        WEIGHT=[]
        THETA=[]
        realtrain=[]
        traindata=[]
        
        if(len(dt)>=1):
            col=len(dt[0])
        buf=[]
        for i in dt:
            if(len(i)>0):
                buf.append(i)
            else:
                traindata.append(buf)
                buf=[]
        traindata.append(buf)
        realdata=[]
        
        for i in traindata:
            onedim=[]
            for j in range(len(i)):
                for k in range(len(i[j])):
                    onedim.append(i[j][k])
            realdata.append(onedim)
        datalen=len(onedim)    
        def hopfield():
            weight=np.zeros((datalen,datalen),dtype=np.float)
            weight=list(weight)
            D=np.eye(datalen,dtype=float)
            D=list(D)
            for i in realdata:
                re=matrixmultidim(i,i)
                #print(re)
                for j in range(len(weight)):
                    for k in range(len(weight[j])):
                        weight[j][k]+=re[j][k]
            for i in range(len(weight)):
                for j in range(len(weight[i])):
                    weight[i][j]=(weight[i][j]/datalen)-D[i][j]*(len(realdata)/datalen)
            
            theta=np.zeros(datalen,dtype=np.float)
            theta=list(theta)
            for i in range(datalen):
                for j in range(len(weight[i])):
                    theta[i]+=weight[i][j]
            return weight,theta        
        WEIGHT,THETA=hopfield()
        realtrain=realdata
        
        
    
    ##########################################plot
    
    
    def presstrain():
        index=int(trainentry.get())
        idx=int(testentry.get())
        drawtrain(idx,index)
        
               
    def drawtrain(idx,index):
        f.clf()
        
        canvas =FigureCanvasTkAgg(f, master=interface)
        
        
       
        a=f.add_subplot(131)
        b=f.add_subplot(132)
        c=f.add_subplot(133)
        for i in range(len(realtrain[index])):
            if(realtrain[index][i]==1):
                
                a.plot(i%colt,int(len(realtrain[index])/colt)-int(i/colt),'bo')         
        for i in range(len(realtest[idx])):
            if(realtest[idx][i]==1):
               
                b.plot(i%colt,int(len(realtest[idx])/colt)-int(i/colt),'bo')
        def recall(ix): 
            accuracy=0
            flag=0
            z=np.zeros(len(realtest[ix]),dtype=np.float)
            z=list(z)
            for i in range(len(WEIGHT)):
                for j in range(len(WEIGHT[i])):
                    z[i]+=WEIGHT[i][j]*realtest[ix][j]
                    
            for k in range(len(z)):
                z[k]=sgn(z[k],z[k])####################################
                    
            while(flag==0):
                flag=1
                z1=np.zeros(len(realtest[ix]),dtype=np.float)
                z1=np.array(z1)
                for i in range(len(WEIGHT)):
                    for j in range(len(WEIGHT)):
                        z1[i]+=WEIGHT[i][j]*z[j]
                        
                for i in range(len(z1)):                                                    
                    z1[i]=sgn(z1[i],z1[i])##################################
                    
                for i in range(len(z1)):
                    if(int(z1[i])!=int(z[i])):
                        #(i)
                        flag=0
                        break
                z=np.copy(z1)
            for i in range(len(z)):
                if(int(z[i])==int(realtest[ix][i])):
                    accuracy+=1
            accuracy=accuracy*100/len(z)
            accuracy=int(accuracy)
            testlabel=tk.Label(interface,text="回想準確率"+str(accuracy)+"%",bg='red',font=('Arial',12),width=15,height=2)
            testlabel.grid(row=0,column=1)
            for i in range(len(z)):
               if(z[i]==1):
                   c.plot(i%colt,int(len(z)/colt)-int(i/colt),'bo')             
        recall(idx)         
        canvas.get_tk_widget().grid(row=7,column=1)
        canvas._tkcanvas.grid(row=7,column=1)  
        
    
         
                
    drawtrainButton=tk.Button(interface,text="draw train",command=presstrain)
    drawtrainButton.grid(row=5,column=0)
    drawtestButton=tk.Button(interface,text="draw test",command=presstrain)
    drawtestButton.grid(row=5,column=1)                    
             
           
        
            
        
    interface.mainloop()
setinter()