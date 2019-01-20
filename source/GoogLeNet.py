from keras.models import Model
from keras.layers import Input
from keras.layers import merge
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

def Inception(in_shape, out_shape):
	input = Input(shape=in_shape)
	conv_1 = Convolution2D(32, (3, 3), use_bias=False)(input)
	BN_1 = BatchNormalization()(conv_1)
	Act_1 = Activation('relu')(BN_1)

	conv_2 = Convolution2D(64, (3, 3), use_bias=False)(Act_1)
	BN_2 = BatchNormalization()(conv_2)
	Act_2 = Activation('relu')(BN_2)
	Pool_1 = MaxPooling2D(pool_size=(2, 2))(Act_2)

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(Pool_1)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(Pool_1)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(Pool_1)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(Pool_1)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)
	
	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(incept_output)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(incept_output)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(incept_output)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(incept_output)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(incept_output)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(incept_output)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(incept_output)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(incept_output)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(incept_output)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(incept_output)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(incept_output)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(incept_output)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)

	conv_3 = Convolution2D(128, (3, 3), use_bias=False)(incept_output)
	BN_3 = BatchNormalization()(conv_3)
	Act_3 = Activation('relu')(BN_3)

	conv_4 = Convolution2D(128, (3, 3), activation='relu')(Act_3)
	Pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)

	conv_5 = Convolution2D(256, (3, 3), activation='relu')(Pool_2)
	conv_6 = Convolution2D(256, (3, 3), activation='relu')(conv_5)

	flat = Flatten()(conv_5)
	DO_1 = Dropout(0.5)(flat)

	Dense_1 = Dense(640, activation='relu')(DO_1)
	DO_2 = Dropout(0.5)(Dense_1)

	Dense_2 = Dense(out_shape, activation='softmax')(DO_2)
	model = Model(input=input, output=Dense_2)

	adam = Adam(lr=0.00001, epsilon=0.001)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
