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
import sys

sys.path.append('/home/inertfluid/.config/spyder-py3/')

from DataPreProcess import LoadData
from ModelCompiler import CompileModel

DataSet = 'KinFaceW-I'
model = CompileModel()
X_Train, Y_Train, X_Test, Y_Test = LoadData(DataSet)

score=[]
for j in range(0, 5):
    X_Train[j] = np.reshape(list(X_Train[j]), (X_Train[j].shape[0], 64, 64, 6))
    Y_Train[j] = keras.utils.to_categorical(Y_Train[j], 2)
    X_Test[j] = np.reshape(list(X_Test[j]), (X_Test[j].shape[0], 64, 64, 6))
    Y_Test[j] = keras.utils.to_categorical(Y_Test[j], 2)
    model.fit(X_Train[j], Y_Train[j], batch_size=64, epochs=50, validation_data=(X_Test[j], Y_Test[j]), shuffle=True)
    score+=[model.evaluate(X_Test[j], Y_Test[j], verbose=1)]
    
score=np.array(score)
loss = np.average(score.T[0])
accuracy = np.average(score.T[1])

print('Loss:', loss)
print('Accuracy:', accuracy)