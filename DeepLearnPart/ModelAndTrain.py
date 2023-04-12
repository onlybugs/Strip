# 22-5-13
# kli
# ....

import torch as t
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import sys
from collections import OrderedDict
import gc

from sklearn.metrics import accuracy_score,precision_score,recall_score,\
    roc_curve,auc,average_precision_score,precision_recall_curve,roc_auc_score
from sklearn.model_selection import KFold

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

lr = 3e-4
pri_n = 800
bz = 20
epoc = 60
decayRate = 0.3
node = sys.argv[1]
cTrain = sys.argv[2]
resb = (3,6,9,9,6,3)

ag = "node:%s\nlr:%f\ndevice:%s\nbatch size:%d\nepoc:%d\nlr_decay:%f\n" % (node,lr,device,bz,epoc,decayRate)

with open(node+".txt",'a+') as f:
    f.write(ag)
    f.close()

print(ag)

def Aupr(l,y):
    aupr = average_precision_score(l,y)
    precision,recall,thresholds = precision_recall_curve(l,y)
    fig = plt.figure(figsize=(8,8))
    plt.title("PR curve")
    plt.xlabel('Recall')
    plt.ylabel("Precision")
    plt.plot(recall,precision,'b',label='AUPR=%0.2f' % aupr)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc = 'lower left')
    plt.close()
    return fig

def roc(l,y):
    fposrate,recall,thresholds = roc_curve(l,y)
    # pre[:,1]

    roc_auc = auc(fposrate,recall)
    fig = plt.figure(figsize=(8,8))
    plt.title('ROC')
    plt.plot(fposrate,recall,'b',label='AUC=%0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('recall')
    plt.xlabel('Fall-out')
    plt.close()
    
    return fig

class MyDataSet(Dataset):
    def __init__(self,k_fold_x,k_fold_y):
        # file = np.load(fp)
        # self.yAE = file['AE']
        # print('dataset init')
        self.X = k_fold_x
        self.yl = k_fold_y
        print(self.yl.shape)
        print(self.yl.sum())

    def __getitem__(self,index):
        
        return self.X[index],self.yl[index]

    def __len__(self):
        
        return self.yl.shape[0]

class ResLayer(nn.Sequential):
    def __init__(self,in_channel,out_channel,ks = 1,p = 0,downsample = None,s = 1):
        super(ResLayer,self).__init__()
        self.add_module("conv1",nn.Conv2d(in_channel,out_channel,kernel_size=ks,padding=p,bias=False,stride=s))
        self.add_module("norm1",nn.BatchNorm2d(out_channel))
        self.add_module("relu1",nn.LeakyReLU())
        self.add_module("conv2",nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,bias=False,stride=1))
        self.add_module("norm2",nn.BatchNorm2d(out_channel))
        # self.drop_rate = drop_rate
        self.downsample = downsample
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        residual = x
        new_features = super(ResLayer,self).forward(x)
        if self.downsample:
            residual = self.downsample(residual)
        new_features += residual

        # relu
        return self.relu(new_features)

class ResBlock(nn.Sequential):
    def __init__(self,num_layers,in_channel,out_channel,ks = 1,p = 0,downsample = None,s = 1):
        super(ResBlock,self).__init__()
        for i in range(num_layers):
            layer = ResLayer(in_channel,out_channel,ks,p,downsample,s)
            self.add_module("reslayer%d" % (i+1),layer)

class ResClass(nn.Module):
    def __init__(self,resb = resb): # 6,9,6
        super(ResClass,self).__init__()
        self.in_channel = 2
        
        self.features = nn.Sequential(OrderedDict([
            ("conv0",nn.Conv2d(2,32,3,padding=1)),
            ("norm0",nn.BatchNorm2d(32)),
            ("relu0",nn.LeakyReLU())
        ]))
        

        in_channel = 32
        for i,layer_num in enumerate(resb):
                block = ResBlock(layer_num,in_channel,in_channel)
                self.features.add_module("resblock%d"%(i+1),block)
                if(i in (0,2,4)):
                    up = self.Upchannel(in_channel,in_channel*2,s = 5)
                    self.features.add_module("up%d"%(i+1),up)
                    in_channel = in_channel * 2

        self.features.add_module("convv",nn.Conv2d(256,128,kernel_size=(1,2),bias=False))
        self.features.add_module("normvv",nn.BatchNorm2d(128))
        self.features.add_module("reluvv",nn.LeakyReLU())

        self.avg = nn.AvgPool2d((5,1))

        self.fc = nn.Sequential(OrderedDict([
            ("fc1",nn.Linear(1024,512)),
            ('fcr1',nn.LeakyReLU()),
            ("fc2",nn.Linear(512,64)),
            ('fcr2',nn.LeakyReLU()),
            ("fc3",nn.Linear(64,2))
        ]))

        self.sigmoid = nn.Sigmoid()

    def Upchannel(self,inc,outc,s = 2):
        return nn.Sequential(nn.Conv2d(inc,outc,3,padding = 1,stride=(s,1),bias=False),
                            nn.BatchNorm2d(outc),
                            nn.LeakyReLU())

    def forward(self,x):
        out = self.features(x)
        out = self.avg(out)
        out = out.view(-1,8*128)
        out = self.fc(out)

        return self.sigmoid(out)

# KFOLD
file = np.load("cv.npz")
# self.yAE = file['AE']
X = file["Sample"]
y = file['label']
kf = KFold(n_splits = 5,shuffle=True,random_state=888)
k = 1

# 5 Fold
for train_index,test_index in kf.split(X):
    avg_auc = 0
    avg_aupr = 0
    num1 = 0

    X = file["Sample"]
    y = file['label']

    traind = MyDataSet(X[train_index],y[train_index])
    validd = MyDataSet(X[test_index],y[test_index])
    valid_loader = DataLoader(validd,batch_size = bz,shuffle=True)
    train_loader = DataLoader(traind,batch_size = bz,shuffle=True)

    # continue
    if(cTrain == "0"):
        modle = ResClass().to(device)
        print("We train model from 0.")
    elif(cTrain == '1'):
        print("We continue train model...\n")
        modle = t.load("smodel/node4-30.pt")

    creti = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(modle.parameters(),lr=lr,weight_decay=5e-4)

    # my_lr_scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    my_lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=50, gamma=decayRate)

    for e in range(epoc):
        lossk = 0
        for num,(X,l) in enumerate(train_loader):
            modle.train()
            X = X.float().to(device)
            # print(l.shape)
            l = l.to(device)
            y = modle(X)
            loss = creti(y,l)
            lossk += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Test
            if(num % pri_n == 0 and num != 0):
                modle.eval()
                with t.no_grad():
                    tloss = 0
                    total_l = None
                    total_yl = None
                    total_yp = None
                    for num2,(X,l) in enumerate(valid_loader):
                        X = X.float().to(device)
                        # print(l.shape)
                        l = l.to(device)
                        y = modle(X)
                        _,pre = t.max(y.data,1)
                        losst = creti(y,l)
                        tloss += losst
                        if(num2 == 0):
                            total_l = l
                            total_yl = pre
                            total_yp = y[:,1]
                        else:
                            total_l = t.hstack((total_l,l))
                            total_yl = t.hstack((total_yl,pre))
                            total_yp = t.hstack((total_yp,y[:,1]))

                    total_l = total_l.cpu().detach()
                    total_yl = total_yl.cpu().detach()
                    total_yp = total_yp.cpu().detach()
                    acc = accuracy_score(total_l,total_yl)
                    pre = precision_score(total_l,total_yl)
                    rec = recall_score(total_l,total_yl)
                    aupr = average_precision_score(total_l,total_yp)
                    auc_ = roc_auc_score(total_l,total_yp)
                    avg_auc += auc_
                    avg_aupr += aupr
                    num1 += 1

                    fig = roc(total_l,total_yp)
                    fig.savefig("cv/plot-cv/" + node + '-' + str(k) + "-" + str(e) + "-" + str(num) + "-roc.png",format = 'png')
                    fig.clear()
                    fig = Aupr(total_l,total_yp)
                    fig.savefig("cv/plot-cv/" + node + '-' + str(k) + "-" + str(e) + "-" + str(num) + "-pr.png",format = 'png')
                    fig.clear()

                    lossk = lossk / pri_n
                    tloss = tloss / num2
                    lst = "Fold %d ,epoch %d ,batchn %d ,loss: %f ,tloss: %f ,acc: %f ,pre: %f ,recall: %f ,auc:%f ,aupr:%f."\
                        % (k,e,num,lossk,tloss,acc,pre,rec,auc_,aupr)
                    print(lst)
                    with open(node+".txt",'a+') as f:
                        f.write(lst+"\n")
                        f.close()

                    lossk = 0
                    total_l = None
                    total_yl = None
                    total_yp = None

        # every epoch save model and valid
        if(e % 20 == 0):
            t.save(modle,"cv/mpath-cv/"+node+ "-" + str(k) + "-" + str(e)+".pt")
        my_lr_scheduler.step()

    s = "cv %d,avg auc %f,avg aupr %f \n" % (k,avg_auc/num1,avg_aupr/num1)
    print(s)
    with open(node+".txt",'a+') as f:
        f.write(s)
        f.close()

    print("cv %d is finish!" % k)
    k += 1

    del traind,validd,modle,creti,optimizer,my_lr_scheduler
    gc.collect()
    t.cuda.empty_cache()
    # bz += 10
    

print("cvDone ^_^")

