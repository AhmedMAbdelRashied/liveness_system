import numpy as np
import cv2
from scipy.signal import filtfilt
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
import scipy
import time
#import psutil
from matplotlib.animation import FuncAnimation
import os
import glob, random
#import skimage.io as io
import tensorflow
from tensorflow import keras



face_cascade = cv2.CascadeClassifier ( r'C:\Users\dell\Desktop\GP_2022\haarcascade_frontalface_default.xml' )


cap = cv2.VideoCapture ( 0 )

model=keras.models.load_model(r'C:\Users\dell\Desktop\new_data\model(7).h5')
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1
        #else:
         #   new_value =0
    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        #new_value = 0
        pass

    return new_value


# Function for calculating LBP
Re=1
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []
    # Now, we need to convert binary
    # values to decimal
    # top_left
    val_ar.append ( get_pixel ( img, center, x - Re, y - Re ) )
    # top
    val_ar.append ( get_pixel ( img, center, x - Re, y ) )
    # top_right
    val_ar.append ( get_pixel ( img, center, x - Re, y + Re ) )

    # right
    val_ar.append ( get_pixel ( img, center, x, y + Re ) )
    # bottom_right
    val_ar.append ( get_pixel ( img, center, x + Re, y + Re ) )

    # bottom
    val_ar.append ( get_pixel ( img, center, x + Re, y ) )

    # bottom_left
    val_ar.append ( get_pixel ( img, center, x + Re, y - Re ) )
    # left
    val_ar.append ( get_pixel ( img, center, x, y - Re ) )


    #power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    #power_val = [128, 64, 32, 16, 8, 4, 2, 1]
    val = 0
    count=0
    for i in range ( len ( val_ar ) ):
        val += val_ar[i] *(2**count) #power_val[i]
      #  print(2 ** count)
        count+=1
    #print(val)
    return val
def LBP(im) :
    height, width= im.shape
    img_lbp = np.zeros ( (height, width), np.uint8 )
    for i in range ( 0, height ):
        for j in range ( 0, width ):
            img_lbp[i, j] = lbp_calculated_pixel ( im, i, j )
           # print(img_lbp[i, j])
   # plt.imshow ( img_lbp, cmap="gray" )
   # plt.show ()
   # print (img_lbp[0])
    return img_lbp
   # print ( "LBP Program is finished" )
dim=(227,227)
while 1:
    ret, img = cap.read ()
    gray = cv2.cvtColor ( img, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale ( gray, 1.3, 5 )
    
    for (x, y, w, h) in faces:
            cv2.rectangle ( img, (x+5, y-5), (x + w-5, y + h+5), (255, 0, 0), 2 )
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            image = cv2.resize(roi_color, dim , interpolation=cv2.INTER_AREA)
            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            Cb =image_YCrCb[:,:,2]
            S =image_HSV[:,:,1]
            V= image_HSV[:,:,2]
            lbp_CB = LBP ( Cb )
            lbp_s = LBP ( S )
            lbp_V = LBP ( V )
            Cb_S_V = cv2.merge ( (lbp_CB, lbp_s, lbp_V) )
            images=[]
            images2=[]
            images2.append(Cb_S_V)
            images=np.array(images2)
            model.predict(images)
            print(model.predict(images))
            #print(np.round(model.predict(images)))
            ind=[]
            ind=np.array(model.predict(images))
           # print(ind)
            a=[]
            a=np.array(ind)
            ind=list(ind.flatten())
            ind=ind.index(np.max(ind))
            print(ind)
            labels = ['real', 'phone','printed','cut']
            font = cv2.FONT_HERSHEY_SIMPLEX
            #org = (00, 185)
            fontScale = 1
            thickness = 2
            #print("labels = " ,labels)
            if (ind==0):
                print("THE image is --> " ,labels[ind])
                img = cv2.putText(img,"real", (x+5, y-5), font, fontScale, (0, 255,0 ), thickness, cv2.LINE_AA, False)
                img = cv2.putText(img,f"{np.ceil(a[0][ind]*100)}%", (x + w, y + h), font, fontScale, (0,255 ,0 ), thickness, cv2.LINE_AA, False)
                cv2.rectangle ( img, (x+5, y-5), (x + w-5, y + h+5), (0, 255,0 ), 2 )
            else :
                print("THE image is -->  fake" )
                img = cv2.putText(img,"fake", (x+5, y-5), font, fontScale, (0,0 ,255 ), thickness, cv2.LINE_AA, False)
                img = cv2.putText(img,f"{np.ceil(a[0][ind]*100)}%", (x + w, y + h), font, fontScale, (0,0 ,255 ), thickness, cv2.LINE_AA, False)
                cv2.rectangle ( img, (x+5, y-5), (x + w-5, y + h+5), (0, 0, 255), 2 )
            
    
            cv2.imshow ( 'img', img )
    k = cv2.waitKey ( 30 ) & 0xff
    if k == 27:
        break

cap.release ()
cv2.destroyAllWindows ()