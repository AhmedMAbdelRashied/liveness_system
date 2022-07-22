#Libraries
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
#cv2.imshow("",x_train[0])
#cv2.waitKey(0)
#images= np.array(images)

y = np.array(y)
import tensorflow
y_train = tensorflow.keras.utils.to_categorical(y)

from tensorflow import keras
import keras
from keras.models import Sequential
# Sequential from keras.models, This gets our neural network as Sequential network.
# As we know, it can be sequential layers or graph
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# Importing, Dense, Activation, Flatten, Activation, Dropout, Conv2D and Maxpool ing.
# Dropout is a technique used to prevent a model from overfitting.
#Instantiate an empty model

#______________________AlexNet CNN Architecture____________________________________
model=Sequential()
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227, 227,3), kernel_size=(11,11), strides=(4,4),padding='valid'))
model.add(Activation('relu'))
# First layer has 96 Filters, the input shape is 227 x 227 x 3
# Kernel Size is 11 x 11, Striding 4 x 4, ReLu is the activation function.
#Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add (Activation('relu'))
# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add (Activation('relu'))
# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
# Passing it to a Fully Connected Layer, Here we do flatten!
model.add(Flatten())
# 1st Fully Connected Layer has 4096 neurons
model.add (Dense(4096, input_shape=(227*227*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout (0.4))
# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout (0.4))
# Output Layer
model.add(Dense(4))
model.add(Activation('softmax'))
#
# Compile the model
#opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
opt=tensorflow.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.001, nesterov=False, name="SGD")

model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train,
                epochs=50 ,
                batch_size=265,
                )
#_______________save Model_______________________
model.save(r'C:\Users\dell\Desktop\GP_TR\model(1).h5')
model.save_weights(r'C:\Users\dell\Desktop\GP_TR\weights_model(1).h5')