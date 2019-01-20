from keras.models import Model
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import merge
import keras.backend as K
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

def AlexNet(in_shape, out_shape):
	input = Input(shape=in_shape)
	lamb1 = Lambda(lambda x: x[:,:,:,0:3], output_shape=(64, 64, 3))(input)
	lamb2 = Lambda(lambda x: x[:,:,:,3:6], output_shape=(64, 64, 3))(input)

	conv_11 = Convolution2D(16, (3, 3))(lamb1)
	BN_11 = BatchNormalization()(conv_11)
	Act_11 = Activation('relu')(BN_11)

	conv_12 = Convolution2D(16, (3, 3))(lamb2)
	BN_12 = BatchNormalization()(conv_12)
	Act_12 = Activation('relu')(BN_12)

	Pool_11 = MaxPooling2D(pool_size=(2, 2))(Act_11)
	Pool_12 = MaxPooling2D(pool_size=(2, 2))(Act_12)

	conv_21_a = Convolution2D(64, (3, 3), activation='relu')(Pool_11)
	conv_21_b = Convolution2D(64, (3, 3), activation='relu')(Pool_12)
	conv_21 = merge([conv_21_a, conv_21_b], mode='concat', concat_axis=3)

	conv_22_a = Convolution2D(64, (3, 3), activation='relu')(Pool_11)
	conv_22_b = Convolution2D(64, (3, 3), activation='relu')(Pool_12)
	conv_22 = merge([conv_22_a, conv_22_b], mode='concat', concat_axis=3)

	conv_31 = Convolution2D(128, (3, 3), activation='relu')(conv_21)
	conv_32 = Convolution2D(128, (3, 3), activation='relu')(conv_22)

	Pool_21 = MaxPooling2D(pool_size=(2, 2))(conv_31)
	Pool_22 = MaxPooling2D(pool_size=(2, 2))(conv_32)

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(Pool_21)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(Pool_21)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(Pool_21)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(Pool_21)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output_1 = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(incept_output_1)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(incept_output_1)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(incept_output_1)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(incept_output_1)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output_1 = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)	

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(incept_output_1)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(incept_output_1)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(incept_output_1)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(incept_output_1)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output_1 = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(Pool_22)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(Pool_22)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(Pool_22)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(Pool_22)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output_2 = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(incept_output_2)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(incept_output_2)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(incept_output_2)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(incept_output_2)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output_2 = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)	

	incept_1x1 = Convolution2D(64, (1, 1), padding='same', activation='relu')(incept_output_2)
	incept_3x3_reduce = Convolution2D(96, (1, 1), padding='same', activation='relu')(incept_output_2)
	incept_3x3 = Convolution2D(128, (3, 3), padding='same', activation='relu')(incept_3x3_reduce)
	incept_5x5_reduce = Convolution2D(16, (1, 1), padding='same', activation='relu')(incept_output_2)
	incept_5x5 = Convolution2D(32, (5, 5), padding='same', activation='relu')(incept_5x5_reduce)
	incept_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(incept_output_2)
	incept_pool_proj = Convolution2D(32, (1, 1), padding='same', activation='relu')(incept_pool)
	incept_output_2 = merge([incept_1x1, incept_3x3, incept_5x5, incept_pool_proj], mode='concat', concat_axis=3)
	
	flat_1 = Flatten()(incept_output_1)
	flat_2 = Flatten()(incept_output_2)

	dense_11_a = Dense(640, activation='relu')(flat_1)
	DO_a = Dropout(0.8)(dense_11_a)
	dense_11_b = Dense(640, activation='relu')(flat_2)
	DO_b = Dropout(0.8)(dense_11_b)
	dense_11 = merge([DO_a, DO_b], mode='concat', concat_axis=1)
	DO_1 = Dropout(0.8)(dense_11)

	dense_12 = merge([DO_a, DO_b], mode='concat', concat_axis=1)
	DO_2 = Dropout(0.75)(dense_12)

	output = merge([DO_1, DO_2], mode='concat', concat_axis=1)
	DO_3 = Dropout(0.75)(output)

	dense_3 = Dense(out_shape, activation='softmax')(output)

	model = Model(input=input, output = dense_3)
	adam = Adam(lr=0.0001, epsilon=0.001, decay=0.001)
	model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
	return model

model = AlexNet((64, 64, 6), 2)
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)