import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import  Counter
import matplotlib.pyplot as plt
#二分类
def plot(im_array):
    print(im_array.shape)
    im_array = im_array*255
    im_array = im_array.astype(int)
    im_array = im_array.reshape(3,9,9)

    print(im_array)

    im_array = im_array.transpose(1,2,0)
    im_array = np.array(im_array)  # 将图片转化为numpy数组
    plt.imshow(im_array)  # 绘制图片
    plt.axis("off")
    plt.savefig("yuan0.png")  # 保存图片
def spilt_trian_test():
    scaler = StandardScaler()
    with open('./processed_data/data_training.npy', 'rb') as fileTrain:
        X  = np.load(fileTrain)

    with open('./processed_data/label_training.npy', 'rb') as fileTrainL:
        Y  = np.load(fileTrainL)
    X = X.reshape(X.shape[0],-1)
    X = normalize(X,norm='max')
    # X = scaler.fit_transform(X)
    # plt.plot(X)
    # plt.show()
    print(X.shape)
    plot(X[300])
    X = X.reshape(-1,3,9,9)
    # X_ = X[0]
    # X_ = X_*255

    Z = np.ravel(Y[:, [1]])
    Z[Z<5]=0
    Z[Z>=5]=1
    number1 =  Counter(Z)
    print(number1)
    y_train = Z
    x_train = np.array(X[:])

    scaler = StandardScaler()
    with open('./processed_data/data_testing.npy', 'rb') as fileTrain:
        M  = np.load(fileTrain)
    with open('./processed_data/label_testing.npy', 'rb') as fileTrainL:
        N  = np.load(fileTrainL)
    M = M.reshape(M.shape[0],-1)
    M = normalize(M,norm='max')
    # M = scaler.fit_transform(M)
    M = M.reshape(-1,3,9,9)
    x_test = np.array(M[:])
    L = np.ravel(N[:, [1]])
    L[L<5]=0
    L[L>=5]=1
    y_test = L
    number2 = Counter(L)
    print(number2)
    x_train,y_train = shuffle(x_train,y_train,random_state=1)
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    return x_train,y_train,x_test,y_test
if __name__ == "__main__":
    spilt_trian_test()