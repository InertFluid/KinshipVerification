from keras.models import Model
from keras.layers import Lambda
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
from keras.models import Sequential

def Nice(in_shape):
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
	return model


def Split(in_shape, out_shape):
	input = Input(shape=in_shape)
	lamb1 = Lambda(lambda x: x[:,:,:,0:3], output_shape=(64, 64, 3))(input)
	lamb2 = Lambda(lambda x: x[:,:,:,3:6], output_shape=(64, 64, 3))(input)

	model = Nice((64, 64, 3))

	output_1 = model(lamb1)
	output_2 = model(lamb2)

	out = merge([output_1, output_2], mode='concat', concat_axis=1)

	dense_2 = Dense(out_shape, activation='softmax')(out)
	split = Model(input=input, output=dense_2)
	adam = Adam(lr=0.001, epsilon=0.001, decay=0.001)
	split.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])	
	return split

model = Split((64, 64, 6), 2)
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)	