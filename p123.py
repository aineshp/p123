from sys import version
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import PIL.ImageOps 
from PIL import Image
import os,ssl,time 
X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']
print(X)
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)
print()    
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=2500,train_size=7500,random_state=9)
xtrainscaled=xtrain/255
xtestscaled=xtest/255
lr=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscaled,ytrain)
ypredict=lr.predict(xtestscaled)
accuracy=accuracy_score(ytest,ypredict)
print(accuracy*100,'%')
capture=cv2.VideoCapture(0)
while(True):
  try:
    ret,frame=capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    height,width=gray.shape
    upper_left=(int(width/2-55)),(int(height/2-55))
    bottom_right=(int(width/2+55)),(int(height/2+55))
    cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
    roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    image_pil=Image.fromarray(roi)
    image_bw=image_pil.convert('L')
    image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
    image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
    pixel_filter=20
    min_pixels=np.percentile(image_bw_resized_inverted,pixel_filter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-min_pixels,0,255)
    max_pixels=np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixels
    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1784)
    test_predict=lr.predict(test_sample)
    print(test_predict)
    cv2.imshow('frame',gray)
    if cv2.WaitKey(1)& 0xFF==ord('q'):
      break
  except Exception as e:
      pass
  capture.release()
  cv2.destroyAllWindows()
  