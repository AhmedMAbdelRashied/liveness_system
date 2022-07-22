from tensorflow import keras
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
package_directory = os.path.dirname(os.path.abspath(__file__))

model1 = keras.models.load_model(os.path.join(package_directory, 'Models/Liveness_model1.h5'))
model2 = keras.models.load_model(os.path.join(package_directory, 'Models/Liveness_model2.h5'))
def get_scores(X):
    X1, X2 = preprocessing(X)
    scores1 = model1.predict(X1)[0,0]
    scores2 = model2.predict(X2)[0]
    sc_d = (scores1 + scores2)/2
    return sc_d

def preprocessing(img):
    image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    S = LBP(image_HSV[:, :, 1])
    V = LBP(image_HSV[:, :, 2])
    Cb = LBP(image_YCrCb[:, :, 2])
    Cb_S_V = cv2.merge((Cb, S, V)).reshape(1, 227, 227, 3)
    S = plt.hist(S.ravel(), 255, [0, 255])[0]
    V = plt.hist(V.ravel(), 255, [0, 255])[0]
    Cb = plt.hist(Cb.ravel(), 255, [0, 255])[0]
    S_V = np.concatenate([S, V], axis=0)
    S_V_Cb = np.concatenate([S_V, Cb], axis=0)
    concat_hist = S_V_Cb.reshape(1, -1)
    return Cb_S_V, concat_hist
    
    
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
if __name__ == '__main__':
    test = cv2.resize(cv2.imread("E:\\Project\\Server\\images\\Ahmed_Rushdi_Mohammed_f1.jpg"), (227, 227))
    preprocessing(test)