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

def CompileModel():
	model=Sequential()
	model.add(Convolution2D(16, (5, 5), input_shape=(64, 64, 6)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, (5, 5)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, (5, 5)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dropout(0.5))
    
	model.add(Dense(640))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Dropout(0.5))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.0001, momentum=0.9, decay=0.005)
	model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])
	print(model.summary())
	return model