import numpy as np
import imageio
import scipy.io
import pandas as pd

def LoadData(DS='KinFaceW-I'):

	count=0
	all_images=[]
	Kin = pd.DataFrame(columns=['Fold', 'Kin/Not-Kin'])

	mat_fd = scipy.io.loadmat(DS + '/meta_data/fd_pairs.mat')
	mat_fd = mat_fd["pairs"]
	mat_fs = scipy.io.loadmat(DS + '/meta_data/fs_pairs.mat')
	mat_fs = mat_fs["pairs"]
	mat_md = scipy.io.loadmat(DS + '/meta_data/md_pairs.mat')
	mat_md = mat_md["pairs"]
	mat_ms = scipy.io.loadmat(DS + '/meta_data/ms_pairs.mat')
	mat_ms = mat_ms["pairs"]

	Mat = [mat_fd, mat_fs, mat_md, mat_ms]
	string = ['father-dau/fd_', 'father-son/fs_', 'mother-dau/md_', 'mother-son/ms_']

	for m in range(0, 4):
		for j in range(0, Mat[m].shape[0]):
			s = DS + '/images/'+ string[m]
			addr = s + Mat[m][j][2][0][3:6]
			image1 = imageio.imread(addr +'_1.jpg')	
			addr = s + Mat[m][j][3][0][3:6]  
			image2 = imageio.imread(addr +'_2.jpg')
			Kin.loc[count] = [Mat[m][j][0][0][0], Mat[m][j][1][0][0]]
			new_image = np.concatenate((image1, image2), axis=2)
			all_images+=[np.array(new_image)]
			count+=1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
	all_images = np.array(all_images)
	all_images = all_images.astype('float32')
	all_images -= np.mean(all_images, axis=0)
	all_images /= np.std(all_images, axis=0)
	Kin = np.array(Kin)
	Data = [all_images, Kin]

	rng_state = np.random.get_state()
	np.random.shuffle(all_images)
	np.random.set_state(rng_state)
	np.random.shuffle(Kin) 

	Folds = [[[], []], [[], []], [[], []], [[], []], [[], []]]
	for i in range (0, all_images.shape[0]):
		for j in range(0, 5):
			if(Data[1][i][0]==j+1):
				Folds[j] = np.append(Folds[j], [[np.array(all_images[i])], [np.array(Kin[i])]], axis=1)

	X=[[], [], [], [], []]
	Y=[[], [], [], [], []] 
	for i in range(0, 5): 
		X[i] = np.array(Folds[i][0])
		Y[i] = np.array([Folds[i][1][j][1] for j in range(Folds[i].shape[1])]) 

	X_Train=[[], [], [], [], []] 
	Y_Train=[[], [], [], [], []]    
	X_Test=[[], [], [], [], []]
	Y_Test=[[], [], [], [], []]    
    
	for i in range(0, 5):    
		X_Train[i] = np.append(X[i%5], X[(i+1)%5], axis=0)
		X_Train[i] = np.append(X_Train[i], X[(i+2)%5], axis=0)
		X_Train[i] = np.append(X_Train[i], X[(i+3)%5], axis=0)
		Y_Train[i] = np.append(Y[i%5], Y[(i+1)%5], axis=0)
		Y_Train[i] = np.append(Y_Train[i], Y[(i+2)%5], axis=0)
		Y_Train[i] = np.append(Y_Train[i], Y[(i+3)%5], axis=0)
		X_Test[i] = X[(i+4)%5]
		Y_Test[i] = Y[(i+4)%5]

	return X_Train, Y_Train, X_Test, Y_Test	