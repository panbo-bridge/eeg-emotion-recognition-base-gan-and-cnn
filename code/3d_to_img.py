from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
with open('./processed_data/s01.npy', 'rb') as fileTrain:
    X  = np.load(fileTrain)
print(X[0][0].shape)
im_array = X[0][0]
    #resolve the python 2 data problem by encoding : latin1
# 使用PIL库和numpy是只是为了快速得到一个可以用于保存为图片的数组，即从现有的图片直接转换成数组
im_array = im_array.transpose(1,2,0)
im_array = np.array(im_array)  # 将图片转化为numpy数组



plt.imshow(im_array)  # 绘制图片
plt.savefig("out_plt2.png")  # 保存图片
