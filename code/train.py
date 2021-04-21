import torch
import numpy as np
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from eeg_loader import spilt_trian_test
import random

from EEGNet import EEGNet_v1,EEGNet_v2,EEGNet_v3,EEGNet_v4,EEGNet_v5
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(1024)
def plot_figure(loss_train,loss_test,acc_test):
    plt.figure(num=1)
    plt.plot(loss_train,"r")
    plt.figure(num=2)
    plt.plot(acc_test,"g")
    plt.show()
def main():
    LR = 0.00001
    x_train,y_train,x_test,y_test = spilt_trian_test()
    batch_size = 64

    device = torch.device("cuda")
    model = EEGNet_v4().to(device)
    # model.load_state_dict(torch.load("best.mdl"))
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    loss_train=[]
    loss_test=[]
    acc_test =[]
    best_acc,best_epoch =0,0
    for epoch in range(200):
        total_train_num = 0
        model.train()
        for i in range((len(x_train)//batch_size-1)+2):#这里列表切片,end超出元素范围也可以
            start = i*batch_size
            end = i*batch_size+batch_size
            x,label = x_train[start:end].to(device),y_train[start:end].to(device)
            logits = model(x)
            loss = criteon(logits,label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_num += x.size(0)
        loss_train.append(loss)
        print("Epoch:{},loss:{:.8f}".format(epoch,loss.item()))

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for i in range((len(x_test)//batch_size-1)+2):
                start = i*batch_size
                end = i*batch_size+batch_size
                x,label = x_test[start:end].to(device),y_test[start:end].to(device)
                logits = model(x)
                loss = criteon(logits,label.long())
                pred = logits.argmax(dim=1)
                correct =torch.eq(pred,label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

            acc = 100.*total_correct/total_num
            loss_test.append(loss)
            acc_test.append(acc)
            print("Epoch:{},accuracy:{:.4f}%".format(epoch,acc))
            if acc > best_acc:
                best_epoch = epoch
                best_acc = acc
                torch.save(model.state_dict(),"best.mdl")
    print("Best Epoch:{},Best accuracy:{:.8f}%".format(best_epoch,best_acc))
    # plot_figure(loss_train,loss_test,acc_test)
if __name__=="__main__":
    main()