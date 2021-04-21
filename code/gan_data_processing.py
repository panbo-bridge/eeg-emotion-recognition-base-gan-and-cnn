import numpy as np
import pickle
import pyeeg as pe
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
band = [4,8,12,16,25,45]
sample_rate = 128
channel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
window_size = 256
step_size = 128
subjectlist = ['01']
list = [4,5,6,7,8,9,10,11,12,13,21,22,27,28,29,32]
def FFT_Processing (sub, channel, band, window_size, step_size, sample_rate):
    meta = []
    with open('./source_data/s' + sub + '.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1') #resolve the python 2 data problem by encoding : latin1

        for i in list:
            # loop over 0-39 trails
            data = subject["data"][i]
            labels = subject["labels"][i]
            # print(labels)
            start = 384;
            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = [] #meta vector for analysis
                for j in channel:
                    X = data[j][start : start + window_size] #Slice raw data over 2 sec, at interval of 0.125 sec
                    Y = pe.bin_power(X, band, sample_rate) #FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                    Y = Y[0]
                    x1,x2,x3 = Y[2],Y[3],Y[4]
                    meta_data.append(x1)
                    meta_data.append(x2)
                    meta_data.append(x3)
                meta_data = np.array(meta_data)
                # print(meta_data.shape)
                meta.append(meta_data)
                start = start + step_size

        meta = np.array(meta)
        # print(meta.shape)
        # print(meta[0].shape)
        np.save('./processed_data/gan/s' + sub, meta, allow_pickle=True, fix_imports=True)
for subjects in subjectlist:
    FFT_Processing (subjects, channel, band, window_size, step_size, sample_rate)
def gan_data():
    with open('./processed_data/gan/s01.npy', 'rb') as fileTrain:
        X  = np.load(fileTrain)
    X = normalize(X,norm='max')
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # plt.plot(X[0])
    # plt.show()
    x_train = np.array(X[:])
    x_train = shuffle(x_train)
    x_train = torch.Tensor(x_train)
    return  x_train
