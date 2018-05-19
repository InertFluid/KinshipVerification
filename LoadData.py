#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 22:00:25 2018

@author: inertfluid
"""

import numpy as np
import imageio
import scipy.io
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization

count=0
all_images=[]
Kin = pd.DataFrame(columns=['Fold', 'Kin/Not-Kin'])

DS = 'KinFaceW-I'

mat_fd = scipy.io.loadmat(DS + '/meta_data/fd_pairs.mat')
mat_fd = mat_fd["pairs"]
mat_fs = scipy.io.loadmat(DS + '/meta_data/fs_pairs.mat')
mat_fs = mat_fs["pairs"]
mat_md = scipy.io.loadmat(DS + '/meta_data/md_pairs.mat')
mat_md = mat_md["pairs"]
mat_ms = scipy.io.loadmat(DS + '/meta_data/ms_pairs.mat')
mat_ms = mat_ms["pairs"]

Mat = [mat_fd, mat_fs, mat_md, mat_ms]
string = ['father-dau/fd_', 'father-son/fs_', 'mother-dau/md_', 'mother-son/ms_']

for m in range(0, 4):
    for j in range(0, Mat[m].shape[0]):
        s = DS + '/images/'+ string[m]
        addr = s + Mat[m][j][2][0][3:6]
        image1 = imageio.imread(addr +'_1.jpg')
        addr = s + Mat[m][j][3][0][3:6]  
        image2 = imageio.imread(addr +'_2.jpg')
        Kin.loc[count] = [Mat[m][j][0][0][0], Mat[m][j][1][0][0]]
        new_image = np.concatenate((image1, image2), axis=2)
        all_images+=[np.array(new_image)]
        count+=1

all_images = np.array(all_images)
all_images = all_images.astype('float32')
all_images -= np.mean(all_images, axis=0)
all_images /= np.std(all_images, axis=0)
Kin = np.array(Kin)
Data = [all_images, Kin]

rng_state = np.random.get_state()
np.random.shuffle(all_images)
np.random.set_state(rng_state)
np.random.shuffle(Kin) 

Fold_1 = [[], []]
Fold_2 = [[], []]
Fold_3 = [[], []]
Fold_4 = [[], []]
Fold_5 = [[], []]
for i in range (0, all_images.shape[0]):
    if(Data[1][i][0]==1):
        Fold_1 = np.append(Fold_1, [[np.array(all_images[i])], [np.array(Kin[i])]], axis=1)
    if(Data[1][i][0]==2):
        Fold_2 = np.append(Fold_2, [[np.array(all_images[i])], [np.array(Kin[i])]], axis=1)
    if(Data[1][i][0]==3):
        Fold_3 = np.append(Fold_3, [[np.array(all_images[i])], [np.array(Kin[i])]], axis=1)
    if(Data[1][i][0]==4):
        Fold_4 = np.append(Fold_4, [[np.array(all_images[i])], [np.array(Kin[i])]], axis=1)
    if(Data[1][i][0]==5):
        Fold_5 = np.append(Fold_5, [[np.array(all_images[i])], [np.array(Kin[i])]], axis=1) 
        
X_1 = np.array(Fold_1[0])
Y_1 = np.array([Fold_1[1][i][1] for i in range(Fold_1.shape[1])]) 
X_2 = np.array(Fold_2[0])
Y_2 = np.array([Fold_2[1][i][1] for i in range(Fold_2.shape[1])]) 
X_3 = np.array(Fold_3[0])
Y_3 = np.array([Fold_3[1][i][1] for i in range(Fold_3.shape[1])]) 
X_4 = np.array(Fold_4[0])
Y_4 = np.array([Fold_4[1][i][1] for i in range(Fold_4.shape[1])]) 
X_5 = np.array(Fold_5[0])
Y_5 = np.array([Fold_5[1][i][1] for i in range(Fold_5.shape[1])])

X_Train_1 = np.append(X_1, X_2, axis=0)
X_Train_1 = np.append(X_Train_1, X_3, axis=0)
X_Train_1 = np.append(X_Train_1, X_4, axis=0)
Y_Train_1 = np.append(Y_1, Y_2, axis=0)
Y_Train_1 = np.append(Y_Train_1, Y_3, axis=0)
Y_Train_1 = np.append(Y_Train_1, Y_4, axis=0)

X_Test_1 = X_5
Y_Test_1 = Y_5

X_Train_2 = np.append(X_2, X_3, axis=0)
X_Train_2 = np.append(X_Train_2, X_4, axis=0)
X_Train_2 = np.append(X_Train_2, X_5, axis=0)
Y_Train_2 = np.append(Y_2, Y_3, axis=0)
Y_Train_2 = np.append(Y_Train_2, Y_4, axis=0)
Y_Train_2 = np.append(Y_Train_2, Y_5, axis=0)

X_Test_2 = X_1
Y_Test_2 = Y_1

X_Train_3 = np.append(X_3, X_4, axis=0)
X_Train_3 = np.append(X_Train_3, X_5, axis=0)
X_Train_3 = np.append(X_Train_3, X_1, axis=0)
Y_Train_3 = np.append(Y_3, Y_4, axis=0)
Y_Train_3 = np.append(Y_Train_3, Y_5, axis=0)
Y_Train_3 = np.append(Y_Train_3, Y_1, axis=0)

X_Test_3 = X_2
Y_Test_3 = Y_2

X_Train_4 = np.append(X_4, X_5, axis=0)
X_Train_4 = np.append(X_Train_4, X_1, axis=0)
X_Train_4 = np.append(X_Train_4, X_2, axis=0)
Y_Train_4 = np.append(Y_4, Y_5, axis=0)
Y_Train_4 = np.append(Y_Train_4, Y_1, axis=0)
Y_Train_4 = np.append(Y_Train_4, Y_2, axis=0)

X_Test_4 = X_3
Y_Test_4 = Y_3

X_Train_5 = np.append(X_5, X_1, axis=0)
X_Train_5 = np.append(X_Train_5, X_2, axis=0)
X_Train_5 = np.append(X_Train_5, X_3, axis=0)
Y_Train_5 = np.append(Y_5, Y_1, axis=0)
Y_Train_5 = np.append(Y_Train_5, Y_2, axis=0)
Y_Train_5 = np.append(Y_Train_5, Y_3, axis=0)

X_Test_5 = X_4
Y_Test_5 = Y_4

WTInit = RandomNormal(mean=0.0, stddev=0.01, seed=5)
model=Sequential()
model.add(Convolution2D(16, (5, 5), input_shape=(64, 64, 6), activation="relu", kernel_initializer=WTInit, bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (5, 5), activation="relu", kernel_initializer=WTInit, bias_initializer="zeros"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (5, 5), activation="relu", kernel_initializer=WTInit, bias_initializer="zeros"))

model.add(Flatten())

model.add(Dense(640, activation="relu", kernel_initializer=WTInit, bias_initializer="zeros"))
model.add(Dense(2, activation="softmax", kernel_initializer=WTInit, bias_initializer="zeros"))

sgd = SGD(lr= 0.01, momentum=0.9, decay=0.005)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
print(model.summary())


X_Train_1 = np.reshape(list(X_Train_1), (X_Train_1.shape[0], 64, 64, 6))
Y_Train_1 = keras.utils.to_categorical(Y_Train_1, 2)
X_Test_1 = np.reshape(list(X_Test_1), (X_Test_1.shape[0], 64, 64, 6))
Y_Test_1 = keras.utils.to_categorical(Y_Test_1, 2)

model.fit(X_Train_1, Y_Train_1, batch_size=50, epochs=50, validation_data=(X_Test_1, Y_Test_1))
score1 = model.evaluate(X_Test_1, Y_Test_1, verbose=0)

X_Train_2 = np.reshape(list(X_Train_2), (X_Train_2.shape[0], 64, 64, 6))
Y_Train_2 = keras.utils.to_categorical(Y_Train_2, 2)
X_Test_2 = np.reshape(list(X_Test_2), (X_Test_2.shape[0], 64, 64, 6))
Y_Test_2 = keras.utils.to_categorical(Y_Test_2, 2)

model.fit(X_Train_2, Y_Train_2, batch_size=50, epochs=50, validation_data=(X_Test_2, Y_Test_2))
score2 = model.evaluate(X_Test_2, Y_Test_2, verbose=0)

X_Train_3 = np.reshape(list(X_Train_3), (X_Train_3.shape[0], 64, 64, 6))
Y_Train_3 = keras.utils.to_categorical(Y_Train_3, 2)
X_Test_3 = np.reshape(list(X_Test_3), (X_Test_3.shape[0], 64, 64, 6))
Y_Test_3 = keras.utils.to_categorical(Y_Test_3, 2)

model.fit(X_Train_3, Y_Train_3, batch_size=50, epochs=50, validation_data=(X_Test_3, Y_Test_3))
score3 = model.evaluate(X_Test_3, Y_Test_3, verbose=0)

X_Train_4 = np.reshape(list(X_Train_4), (X_Train_4.shape[0], 64, 64, 6))
Y_Train_4 = keras.utils.to_categorical(Y_Train_4, 2)
X_Test_4 = np.reshape(list(X_Test_4), (X_Test_4.shape[0], 64, 64, 6))
Y_Test_4 = keras.utils.to_categorical(Y_Test_4, 2)

model.fit(X_Train_4, Y_Train_4, batch_size=50, epochs=50, validation_data=(X_Test_4, Y_Test_4))
score4 = model.evaluate(X_Test_4, Y_Test_4, verbose=0)

X_Train_5 = np.reshape(list(X_Train_5), (X_Train_5.shape[0], 64, 64, 6))
Y_Train_5 = keras.utils.to_categorical(Y_Train_5, 2)
X_Test_5 = np.reshape(list(X_Test_5), (X_Test_5.shape[0], 64, 64, 6))
Y_Test_5 = keras.utils.to_categorical(Y_Test_5, 2)

model.fit(X_Train_5, Y_Train_5, batch_size=50, epochs=50, validation_data=(X_Test_5, Y_Test_5))
score5 = model.evaluate(X_Test_5, Y_Test_5, verbose=0)

score5
loss = score1[0] + score2[0] + score3[0] + score4[0] + score5[0]
loss/= 5

accuracy = score1[1] + score2[1] + score3[1] + score4[1] + score5[1]
accuracy/=5
exit()
