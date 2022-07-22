#_________________Libraries________________________
import cv2
import numpy as np
import os
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
#import images

#__________________Function to get pixel__________________________
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

#__________Function for calculating LBP_____________________
Re=1 #Radius for LBP
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

#___________Function for calculating LBP and return image after LBP____________________
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

images = []
y_pred=[]
y2 = []
H = []
S = []
V = []
Y = []
Cb = []
Cr = []
dim=(227,227)
face_cascade = cv2.CascadeClassifier ( r'C:\Users\dell\Desktop\GP_2022\haarcascade_frontalface_default.xml' )
model=keras.models.load_model(r'C:\Users\dell\Desktop\new_data\model(7).h5')

labels = ['real', 'phone','printed','cut']
#data = []
y=[]
y_pred=[]
#images=[]
dim=(227,227)

#___________Read and Predict LBP Images____________________
for i in range(1,5001):
    try:
        #image = cv2.imread ( rf'C:\Users\dell\Desktop\new_data\r_e_a_l\{i}.jpg' )
        image = cv2.imread ( rf'C:\Users\dell\Desktop\GP_2022\rrr\r({i}).jpg' )
        image = cv2.resize ( image, dim, interpolation=cv2.INTER_AREA )
        images = []
        images2 = []
        images2.append(image)
        images = np.array ( images2 )
        q2 = model.predict ( images )
        ind = np.array ( q2 )
        ind = list ( ind.flatten () )
        ind = ind.index ( np.max ( ind ) )
        # print(ind)
        y_pred.append ( ind )
        y.append(0)

        #image = cv2.imread (  rf'C:\Users\dell\Desktop\new_data\f_a_k_e\{i}.jpg')
        image = cv2.imread ( rf'C:\Users\dell\Desktop\GP_2022\vvv\v({i}).jpg' )
        image = cv2.resize ( image, dim, interpolation=cv2.INTER_AREA )
        images = []
        images2 = []
        images2.append ( image )
        images = np.array ( images2 )
        q2 = model.predict ( images )
        ind = np.array ( q2 )
        ind = list ( ind.flatten () )
        ind = ind.index ( np.max ( ind ) )
        # print(ind)
        y_pred.append ( ind )
        y.append(1)

        image = cv2.imread( rf'C:\Users\dell\Desktop\GP_2022\ppp\p({i}).jpg')
        image = cv2.resize ( image, dim, interpolation=cv2.INTER_AREA )
        images = []
        images2 = []
        images2.append ( image )
        images = np.array ( images2 )
        q2 = model.predict ( images )
        ind = np.array ( q2 )
        ind = list ( ind.flatten () )
        ind = ind.index ( np.max ( ind ) )
        # print(ind)
        y_pred.append ( ind )
        y.append ( 2 )

        image = cv2.imread (rf'C:\Users\dell\Desktop\GP_2022\ccc\c({i}).jpg' )
        image = cv2.resize ( image, dim, interpolation=cv2.INTER_AREA )
        images = []
        images2 = []
        images2.append ( image )
        images = np.array ( images2 )
        q2 = model.predict ( images )
        ind = np.array ( q2 )
        ind = list ( ind.flatten () )
        ind = ind.index ( np.max ( ind ) )
        # print(ind)
        y_pred.append ( ind )
        y.append ( 3 )
        print ( i )
    except :
        pass

    #print(i)

#___________processing for y_test and  y_predict________________
y = np.array(y)
y_pred=np.array(y_pred)
y2=[]
for k in range(len(y)):
    if(y[k]==0):
        y2.append(0)
    else:
        y2.append(1)
y_pred2=[]
for m in range(len(y_pred)):
    if(y_pred[m]==0):
        y_pred2.append(0)
    else:
        y_pred2.append(1)
y2 = np.array(y2)
y_pred2=np.array(y_pred2)

#__________labels________________
labels = ['Real', 'Phone','Printed','Cut']
#classification_report for the 4 class or labels using y_test and  y_predict
from sklearn.metrics import classification_report
print("Test result on CASIA-FASD dataset")
#print("Test result on MSU-MFSD dataset")
#classification_report for the 2 class or labels using y_test and  y_predict
print(classification_report(y, y_pred, target_names=labels))
labels2 = ['Real', 'Attack']
print("Test result on CASIA-FASD dataset")
#print("Test result on MSU-MFSD dataset")
print(classification_report(y2, y_pred2, target_names=labels2))






