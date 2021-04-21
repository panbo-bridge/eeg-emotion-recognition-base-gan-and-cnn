import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from gan_data_processing import  gan_data
import matplotlib.pyplot as plt
batch_size = 128
num_epoch = 600
z_dimension = 16
num_sample = 464
device = torch.device("cuda")
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(96,64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.dis(x)
        return x
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 96),
            nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


def main():
    torch.manual_seed(55)
    np.random.seed(55)
    x_train = gan_data()
    D = discriminator()
    G = generator()
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()
    criterion = nn.BCELoss()#使用这个函数前一般会加上sigmoid函数
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
    best_min_loss = 1
    for epoch in range(num_epoch):
        for i in range((len(x_train)//batch_size-1)+2):#这里列表切片,end超出元素范围也可以
            start = i*batch_size
            end = i*batch_size+batch_size
            real_x = x_train[start:end].to(device)
            # print(real_x.shape)
            real_label = Variable(torch.ones(real_x.shape[0])).cuda()
            fake_label = Variable(torch.zeros(real_x.shape[0])).cuda()
            #训练判别器
            real_out = D(real_x)
            # print(real_out.shape)
            # print(real_label.shape)
            d_loss_real = criterion(real_out,real_label)
            real_scores = real_out

            z = Variable(torch.randn(real_x.shape[0],z_dimension)).cuda()
            fake_x = G(z)
            fake_out = D(fake_x)
            d_loss_fake = criterion(fake_out,fake_label)
            fake_scores = fake_out

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            #训练生成器
            z = Variable(torch.randn(real_x.shape[0],z_dimension)).cuda()
            fake_x = G(z)
            output = D(fake_x)
            g_loss = criterion(output,real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            #输出
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                      epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                      real_scores.data.mean(), fake_scores.data.mean()))

    z = Variable(torch.randn(num_sample,z_dimension)).cuda()
    fake_x =  G(z)
    fake_x = fake_x.cpu().data
    np.save('./processed_data/s' + "01_fake",fake_x, allow_pickle=True, fix_imports=True)
    print(fake_x.shape)
    fake_x = fake_x.cpu().data
    real_x = real_x.cpu().data
    # print(type(fake_x[0]))
    plt.figure(num=1)
    plt.plot(fake_x[0],"r")
    plt.figure(num=2)
    plt.plot(real_x[0],"g")
    plt.show()

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
def fake_fame():
    meta =  []
    with open('./processed_data/s' + "01_fake" + '.npy', 'rb') as file:
        sub = np.load(file, allow_pickle=True)
        sub = sub.reshape(num_sample,32,3)
        labels = np.array([0,1,0,0])
        for i in range(num_sample):
            meta_array = []
            meta_data = []
            x1_2D = []
            x2_2D = []
            x3_2D = []
            for j in range(32):
                x1_2D.append(sub[i][j][0])
                x2_2D.append(sub[i][j][1])
                x3_2D.append(sub[i][j][2])
            x1_2D = data_1Dto2D(x1_2D)
            x2_2D = data_1Dto2D(x2_2D)
            x3_2D = data_1Dto2D(x3_2D)
            x1_2D = np.array(x1_2D)
            x2_2D = np.array(x2_2D)
            x3_2D = np.array(x3_2D)
            meta_data.append(x1_2D)
            meta_data.append(x2_2D)
            meta_data.append(x3_2D)
            meta_data = np.array(meta_data)
            meta_array.append(meta_data)
            meta_array.append(labels)
            meta.append(np.array(meta_array))
        meta = np.array(meta)
        print(meta.shape)
        print(meta[0][0].shape)
        np.save('./processed_data/s' + "003", meta, allow_pickle=True, fix_imports=True)
        # print(sub.shape)

if __name__ ==  "__main__":
    # main()
    fake_fame()