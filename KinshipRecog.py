# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Convolution3D
from keras.layers import MaxPooling3D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomNormal

WTInit = RandomNormal(mean=0.0, stddev=0.01, seed=5)
model = Sequential()
model.add(Convolution3D(16, (5, 5, 1), input_shape=(64, 64, 1, 6), kernel_initializer=WTInit, bias_initializer="zeros"))
model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1)))

model.add(Convolution3D(64, (5, 5, 1), activation="relu", kernel_initializer=WTInit, bias_initializer="zeros"))
model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1)))

model.add(Convolution3D(128, (5, 5, 1), activation="relu", kernel_initializer=WTInit, bias_initializer="zeros"))

model.add(Flatten())

model.add(Dense(units=640, activation="relu", kernel_initializer=WTInit, bias_initializer="zeros" ))

model.add(Dense(units=2, activation="softmax", kernel_initializer=WTInit, bias_initializer="zeros"))

sgd = SGD(lr= 0.01, momentum=0.9, decay=0.005)
model.compile(optimizer=sgd, loss="categorical_crossentropy")

print(model.summary())