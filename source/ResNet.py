from keras.models import Model
from keras import layers
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import merge
import keras.backend as K
from keras.layers import ZeroPadding2D
from keras.regularizers import l2
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
from keras.layers import AveragePooling2D
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D

def ResNet18(in_shape, out_shape):
	input = Input(shape=in_shape)
	conv_5x5 = Convolution2D(64, (7,7), use_bias=False)(input)
	BN_1 = BatchNormalization(axis=3)(conv_5x5)
	Act_1 = Activation('relu')(BN_1)
	ZeroPad = ZeroPadding2D(padding=((1, 1), (1, 1)))(Act_1)
	x = MaxPooling2D(3, strides=1)(ZeroPad)

	for i in range(0, 4):
		input_tensor=x
		x = Convolution2D(64*(2**i), (3, 3), padding='same', strides=2)(x)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)
		x = Convolution2D(64*(2**i), (3, 3), padding='same')(x)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)		
		shortcut = Convolution2D(64*(2**i), (1, 1), strides=2)(input_tensor)
		shortcut = BatchNormalization(axis=3)(shortcut)
		x = layers.add([x, shortcut])
		x = Activation('relu')(x)

		input_tensor=x
		x = Convolution2D(64*(2**i), (3, 3), padding='same')(x)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)
		x = Convolution2D(64*(2**i), (3, 3), padding='same')(x)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)		
		x = layers.add([x, input_tensor])
		x = Activation('relu')(x)

	x = AveragePooling2D((3, 3))(x)
	x = Flatten()(x)
	x = Dense(out_shape, activation='softmax')(x)

	model = Model(input, x)
	sgd = SGD(lr=0.001, momentum=0.9, decay=0.0001)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
	return model

model = ResNet18((64, 64, 6), 2)
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)	


