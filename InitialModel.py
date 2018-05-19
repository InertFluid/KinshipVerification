from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomNormal

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
model.compile(optimizer=sgd, loss="categorical_crossentropy")
print(model.summary())
