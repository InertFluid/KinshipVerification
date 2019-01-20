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

def VGG_16(in_shape, out_shape):
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

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(out_shape, name='dense_3'))
    model.add(Activation('softmax', name='softmax'))

    sgd = SGD(lr=0.0001, momentum=0.9, decay=0.005)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

