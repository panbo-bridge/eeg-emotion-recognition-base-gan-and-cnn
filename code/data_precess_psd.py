import numpy as np
import pickle
import pyeeg as pe
band = [4,8,12,16,25,45]
sample_rate = 128
channel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
window_size = 256
step_size = 128
subjectlist_1 = ['01']
subjectlist_2 = ["07","003"]
# list = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],[4,5,6,9,10,12,16,20]]
# list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
list = [4,5,6,7,8,9,10,11,12,13,21,22,27,28,29,32]
def data_1Dto2D(data, Y=9, X=9):
	data_2D = np.zeros([Y, X])
	data_2D[0] = (0,        0,          0,          data[0],    0,          data[16],   0,          0,          0       )
	data_2D[1] = (0,        0,          0,          data[1],    0,          data[17],   0,          0,          0       )
	data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
	data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
	data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
	data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
	data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
	data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
	data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
	return data_2D

def FFT_Processing (sub, channel, band, window_size, step_size, sample_rate):
    meta = []
    with open('./source_data/s' + sub + '.dat', 'rb') as file:
        # seq = int(sub)-1
        subject = pickle.load(file, encoding='latin1') #resolve the python 2 data problem by encoding : latin1

        for i in list:
            # loop over 0-39 trails
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 384;
            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = [] #meta vector for analysis
                x1_2d=[]
                x2_2d=[]
                x3_2d=[]
                for j in channel:
                    X = data[j][start : start + window_size] #Slice raw data over 2 sec, at interval of 0.125 sec
                    Y = pe.bin_power(X, band, sample_rate) #FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                    Y = Y[0]
                    x1,x2,x3= Y[2],Y[3],Y[4]
                    x1_2d.append(x1)
                    x2_2d.append(x2)
                    x3_2d.append(x3)
                x1_2d = data_1Dto2D(x1_2d,9,9)
                x2_2d = data_1Dto2D(x2_2d,9,9)
                x3_2d = data_1Dto2D(x3_2d,9,9)
                meta_data.append(x1_2d)
                meta_data.append(x2_2d)
                meta_data.append(x3_2d)
                meta_data = np.array(meta_data)
                meta_array.append(meta_data)
                meta_array.append(labels)
                meta.append(np.array(meta_array))
                start = start + step_size

        meta = np.array(meta)
        print(meta.shape)
        print(meta[0][0].shape)
        np.save('./processed_data/s' + sub, meta, allow_pickle=True, fix_imports=True)
# for subjects in subjectlist_1:
#     FFT_Processing (subjects, channel, band, window_size, step_size, sample_rate)
# FFT_Processing ('01', channel, band, window_size, step_size, sample_rate)
def spilt_train_test():
    data_training = []
    label_training = []
    data_testing = []
    label_testing = []
    for subjects in subjectlist_1:
        with open('./processed_data/s' + subjects + '.npy', 'rb') as file:
            sub = np.load(file,allow_pickle=True)
            for i in range(0,sub.shape[0]):
                if i % 8 == 0:
                    data_testing.append(sub[i][0])
                    label_testing.append(sub[i][1])
                else:
                    data_training.append(sub[i][0])
                    label_training.append(sub[i][1])

    np.save('./processed_data/data_training', np.array(data_training), allow_pickle=True, fix_imports=True)
    np.save('./processed_data/label_training', np.array(label_training), allow_pickle=True, fix_imports=True)
    print("training dataset:", np.array(data_training).shape, np.array(label_training).shape)

    np.save('./processed_data/data_testing', np.array(data_testing), allow_pickle=True, fix_imports=True)
    np.save('./processed_data/label_testing', np.array(label_testing), allow_pickle=True, fix_imports=True)
    print("testing dataset:", np.array(data_testing).shape, np.array(label_testing).shape)
spilt_train_test()