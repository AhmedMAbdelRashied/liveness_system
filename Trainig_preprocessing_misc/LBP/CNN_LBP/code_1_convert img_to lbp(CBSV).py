#_________________Libraries________________________
import cv2.cv2
import numpy as np
import os
#Function to get pixel
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
    return img_lbp


y = []
H = []
S = []
V = []
Y = []
Cb = []
Cr = []
dim=(227,227)
labels = ['real', 'phone','printed','cut']
data = []

#____Read the images and then resize them to (227 and 227)__________
#____then converting them to YCbCr and HSV color spaces,____________
#____then apply the LBP after that save them in a file______________
for i in range(0,16000):
   # image = cv2.imread(rf'C:\Users\dell\Desktop\pythonProject\test\New folder\RAEL\R({i}).jpg')
    image = cv2.imread ( rf'E:\desktop_for_gproj_ 14_3_2022\pythonProject\test\New folder\RAEL\R({i+484+1}).jpg' )
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_HSV = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2HSV)
    image_YCrCb = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2YCrCb)
    S = image_HSV[:, :, 1]
    V = image_HSV[:, :, 2]
    Cb = image_YCrCb[:, :, 2]
    lbp_CB = LBP ( Cb )
    lbp_s = LBP ( S )
    lbp_V = LBP ( V )
    Cb_S_V = cv2.cv2.merge ( (lbp_CB, lbp_s, lbp_V) )
    cv2.imwrite( rf"C:\Users\dell\Desktop\GP_2022\r\{i}.jpg",Cb_S_V )
   # H_S_V_Y_Cr_Cb = cv2.cv2.merge((image_HSV[:,:,0],image_HSV[:,:,1],image_HSV[:,:,2],image_YCrCb[:,:,0],image_YCrCb[:,:,1],image_YCrCb[:,:,2]))
    #images.append(Cb_S_V)

    #y.append(0)

    #image = cv2.imread(rf'C:\Users\dell\Desktop\pythonProject\test\New folder\VIDEO\F({i}).jpg')
    image = cv2.imread ( rf'E:\desktop_for_gproj_ 14_3_2022\pythonProject\test\New folder\VIDEO\F({i+3601}).jpg' )
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_HSV = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2HSV)
    image_YCrCb = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2YCrCb)
    S = image_HSV[:, :, 1]
    V = image_HSV[:, :, 2]
    Cb = image_YCrCb[:, :, 2]
    lbp_CB = LBP ( Cb )
    lbp_s = LBP ( S )
    lbp_V = LBP ( V )
    Cb_S_V = cv2.cv2.merge ( (lbp_CB, lbp_s, lbp_V) )
    # H_S_V_Y_Cr_Cb = cv2.cv2.merge((image_HSV[:,:,0],image_HSV[:,:,1],image_HSV[:,:,2],image_YCrCb[:,:,0],image_YCrCb[:,:,1],image_YCrCb[:,:,2]))
    cv2.imwrite(rf"C:\Users\dell\Desktop\GP_2022\v\{i}.jpg",Cb_S_V)
    i#mages.append ( Cb_S_V )
    #y.append(1)

    #image = cv2.imread(rf'C:\Users\dell\Desktop\pythonProject\test\New folder\PRINT\F({i}).jpg')
    image = cv2.imread(rf'E:\desktop_for_gproj_ 14_3_2022\pythonProject\test\New folder\PRINT\F({i+6935}).jpg')
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_HSV = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2HSV)
    image_YCrCb = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2YCrCb)
    S = image_HSV[:, :, 1]
    V = image_HSV[:, :, 2]
    Cb = image_YCrCb[:, :, 2]
    lbp_CB = LBP ( Cb )
    lbp_s = LBP ( S )
    lbp_V = LBP ( V )
    Cb_S_V = cv2.cv2.merge ( (lbp_CB, lbp_s, lbp_V) )
    cv2.imwrite( rf"C:\Users\dell\Desktop\GP_2022\p\{i}.jpg",Cb_S_V )
    # H_S_V_Y_Cr_Cb = cv2.cv2.merge((image_HSV[:,:,0],image_HSV[:,:,1],image_HSV[:,:,2],image_YCrCb[:,:,0],image_YCrCb[:,:,1],image_YCrCb[:,:,2]))
    #images.append ( Cb_S_V )
    #y.append(2)

    #image = cv2.imread(rf'C:\Users\dell\Desktop\pythonProject\test\New folder\CUT\F({i}).jpg')
    image = cv2.imread ( rf'E:\desktop_for_gproj_ 14_3_2022\pythonProject\test\New folder\CUT\F({i}).jpg' )

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_HSV = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2HSV)
    image_YCrCb = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2YCrCb)
    S = image_HSV[:, :, 1]
    V = image_HSV[:, :, 2]
    Cb = image_YCrCb[:, :, 2]
    lbp_CB = LBP ( Cb )
    lbp_s = LBP ( S )
    lbp_V = LBP ( V )
    Cb_S_V = cv2.cv2.merge ( (lbp_CB, lbp_s, lbp_V) )
    # H_S_V_Y_Cr_Cb = cv2.cv2.merge((image_HSV[:,:,0],image_HSV[:,:,1],image_HSV[:,:,2],image_YCrCb[:,:,0],image_YCrCb[:,:,1],image_YCrCb[:,:,2]))
    cv2.imwrite( rf"C:\Users\dell\Desktop\GP_2022\c\{i}.jpg",Cb_S_V )
    #images.append ( Cb_S_V )
    #y.append(3)
    print(i+1)
