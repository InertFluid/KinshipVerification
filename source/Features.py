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

def CornerFeatures():
	input1 = Input(shape=(39, 39, 6))
	input2 = Input(shape=(39, 39, 6))
	input3 = Input(shape=(39, 39, 6))
	input4 = Input(shape=(39, 39, 6))
	input5 = Input(shape=(39, 39, 6))

	from ModelCompiler import CompileModel
	model1 = CompileModel((39, 39, 6), 2)
	model2 = CompileModel((39, 39, 6), 2)
	model3 = CompileModel((39, 39, 6), 2)
	model4 = CompileModel((39, 39, 6), 2)
	model5 = CompileModel((39, 39, 6), 2)

	output_1 = model1(input1)
	output_2 = model2(input2)
	output_3 = model3(input3)
	output_4 = model4(input4)
	output_5 = model5(input5)

	out = merge([output_1, output_2, output_3, output_4, output_5], mode='concat', concat_axis=1)

	dense_1 = Dense(640, activation='relu')(out)
	DO_1 = Dropout(0.75)(dense_1)

	dense_2 = Dense(2, activation='softmax')(DO_1)
	split = Model(input=[input1, input2, input3, input4, input5], output=dense_2)
	adam = Adam(lr=0.001, epsilon=0.001, decay=0.001)
	split.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])	
	return split	

