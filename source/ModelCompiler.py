from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization

def CompileModel(in_shape, out_shape):
	print("Compiling Model...")
	model=Sequential()
	model.add(Convolution2D(32, (3, 3), input_shape=in_shape, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, (3, 3), use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(64, (3, 3), use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(Convolution2D(256, (3, 3)))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dropout(0.5))
    
	model.add(Dense(640))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))
	model.add(Dense(out_shape))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.000001, momentum=0.9,decay=.001)
	adam = Adam(lr=0.0001, epsilon=0.001, decay=0.001)
	model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
	return model
	