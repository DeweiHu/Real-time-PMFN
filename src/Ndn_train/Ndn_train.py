import sys
sys.path.insert(0,'E:\\real-time-PMFN\\data\\')
import NetworkArch
import util

import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

global root
root = 'E:\\Retina2_ONH\\'
modelroot = 'E:\\Model\\'

def Mosaic(pattern,*args):
    [nr,nc] = pattern
    [r,c] = args[0].shape
    grand = np.zeros([nr*r,nc*c],dtype=np.float32)
    
    cnt = 0
    for im in args:
        idx_r = int(np.floor(cnt/nc))
        idx_c = cnt % nc
        grand[idx_r*r:(idx_r+1)*r,idx_c*c:(idx_c+1)*c] = util.ImageRescale(im,[0,255])
        cnt += 1
    return grand
        
#%%
print('Creating dataset...')

batch_size = 1

class MyDataset(Data.Dataset):

    def ToTensor(self,x,y1,y2):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        y1_tensor = torch.tensor(y1).type(torch.FloatTensor)
        y1_tensor = torch.unsqueeze(y1_tensor,dim=0)
        y2_tensor = torch.tensor(y2).type(torch.FloatTensor)
        y2_tensor = torch.unsqueeze(y2_tensor,dim=0)
        return x_tensor, y1_tensor, y2_tensor    
        
    def __init__(self, dataroot):
        with open(dataroot,'rb') as f:
            self.pair = pickle.load(f)
        
    def __len__(self):
        return len(self.pair)

    def __getitem__(self,idx):
        (x,y1,y2) = self.pair[idx]
        # only load the original noisy image channel
        x_tensor, y1_tensor, y2_tensor = self.ToTensor(x,util.ImageRescale(y1,[0,1]),
                                            util.ImageRescale(y2,[0,1]))
        return x_tensor, y1_tensor, y2_tensor
    
train_loader = Data.DataLoader(dataset=MyDataset(root+'dual_train.pickle'), 
                               batch_size=batch_size, shuffle=True)

#%% Loading Network
print('Initializing model...')

gpu = 1
device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")
in_channels = 3

# load the training network 
Ndn = NetworkArch.MS_UNet(gpu,in_channels).to(device)
Ndn.apply(NetworkArch.weight_init)

#%%
L1_loss = nn.L1Loss()
L2_loss = nn.MSELoss()

beta1 = 0.5
lr = 1e-4
a = 1
b = 2

optimizer = optim.Adam(Ndn.parameters(),lr=lr,betas=(beta1,0.999))
scheduler = StepLR(optimizer, step_size=1, gamma=0.3)

#%%
print('Training start...')

import time

num_epoch = 15

t1 = time.time()
for epoch in range(num_epoch):
    for step,[x,y1,y2] in enumerate(train_loader):
        
        Ndn.train()
        
        train_x = Variable(x).to(device)        
        train_y1 = Variable(y1).to(device)
        train_y2 = Variable(y2).to(device)
        pred = Ndn(train_x)
                
        loss_1 = a*L1_loss(pred,train_y1)
        loss_2 = b*L2_loss(pred,train_y2)
        loss = loss_1+loss_2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_1:%.4f\tLoss_2:%.4f'
                  %(epoch, num_epoch, step, len(train_loader),loss_1,loss_2))
            
        if step % 500 == 0:
            with torch.no_grad():
                
                im_grand = np.zeros([1024,1000],dtype=np.float32)
                
                ipt = x[0,:,:,:500].numpy()
                dnoi = pred[0,0,:,:500].detach().cpu().numpy()
                avg = y1[0,0,:,:500].numpy()
                sf = y2[0,0,:,:500].numpy()
                
                grand = Mosaic([2,3],ipt[0,:,:],ipt[2,:,:],ipt[1,:,:],
                               avg,sf,dnoi)
                
                plt.figure(figsize=(15,10))
                plt.axis('off')
                plt.imshow(grand,cmap='gray')
                plt.show()
    scheduler.step()

#%%
name = '1_2.pt'
torch.save(Ndn.state_dict(),modelroot+name)
