#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 07:58:17 2018

@author: inertfluid
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization

in_shape = (224, 224, 3)
out_shape = 1000

model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape=in_shape, activation='relu', name='conv1_1'))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='dense_1'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='dense_2'))
model.add(Dropout(0.5))
model.add(Dense(out_shape, name='dense_3'))
model.add(Activation('softmax', name='softmax'))

model.summary()