from keras.models import Model
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

def transition_block(x, reduction=0.5):
	x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
	x = Activation('relu')(x)
	x = Convolution2D(int(K.int_shape(x)[3] * reduction),1,use_bias=False)(x)
	x = AveragePooling2D(2, strides=2)(x)
	return x


def conv_block(x, growth_rate=32):
  x1 = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
  x1 = Activation('relu')(x1)
  x1 = Convolution2D(4*growth_rate, 1, use_bias=False)(x1)
  x1 = BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
  x1 = Activation('relu')(x1)
  x1 = Convolution2D(growth_rate, 3, padding='same', use_bias=False)(x1)
  x = Concatenate(axis=3)([x, x1])
  return x

def DenseNet(in_shape, out_shape, blocks):
	input = Input(shape=(64, 64, 6))
	conv_7x7 = Convolution2D(64, (7,7), use_bias=False)(input) #strides 1 instead of 2
	BN_1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv_5x5)
	Act_1 = Activation('relu')(BN_1)
	ZeroPad = ZeroPadding2D(padding=((1, 1), (1, 1)))(Act_1)
	x = MaxPooling2D(3, strides=1)(ZeroPad) #strides 1 instead of 2

	for i in range(len(blocks)):
		for j in range(blocks[i]):
			x=conv_block(x)
		if i != len(blocks)-1:
			x=transition_block(x)

	x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(out_shape, activation='softmax')(x)
	model = Model(input, x)
	sgd = SGD(lr=0.001, momentum=0.9, decay=0.0001)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
	return model	  	

model = DenseNet((64, 64, 6), 2, [2, 2, 2, 2])
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)	
