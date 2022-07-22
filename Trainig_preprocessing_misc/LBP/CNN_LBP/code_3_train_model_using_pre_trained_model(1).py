import cv2
import numpy as np
import os
#import images
images = []
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
#_______________Read LBP images______________
for i in range(0,5000):
    image = cv2.imread ( rf'C:\Users\dell\Desktop\GP_2022\rr\R({i+1}).jpg' )
    images.append(image)
    y.append(0)

    image = cv2.imread (  rf'C:\Users\dell\Desktop\GP_2022\vv\v({i+1}).jpg')
    images.append(image)

    y.append(1)

    image = cv2.imread( rf'C:\Users\dell\Desktop\GP_2022\pp\P({i+1}).jpg')
    images.append ( image )
    y.append(2)

    image = cv2.imread (  rf'C:\Users\dell\Desktop\GP_2022\cc\c({i+1}).jpg' )
    images.append(image)
    y.append(3)

    print(i+1)

x_train = np.array(images)
y = np.array(y)
import tensorflow
y_train = tensorflow.keras.utils.to_categorical(y)
from tensorflow import keras
import keras
from keras.models import Sequential
# Sequential from keras.models, This gets our neural network as Sequential network.
# As we know, it can be sequential layers or graph
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

#______________Load Model______________________

model=keras.models.load_model(r'C:\Users\dell\Desktop\GP_2022\model(2).h5')
#opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
opt=tensorflow.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.001, nesterov=False, name="SGD")
#______________train Model______________________
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train,
                epochs=50 ,
                batch_size=265,
                 )
#______________Save Model______________________
model.save(r'C:\Users\dell\Desktop\GP_2022\model(7).h5')
model.save_weights(r'C:\Users\dell\Desktop\GP_2022\weights_model(7).h5')