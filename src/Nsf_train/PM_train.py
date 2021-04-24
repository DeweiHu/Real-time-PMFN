import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import util
import Arch

import numpy as np
import time, pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

global nch 
nch = 3
gpu = 1
nepoch = 50
dataroot = 'E:\\HumanData\\PM_traindata.pickle'
modelroot = 'E:\\Model\\'
device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")
    
class MyDataset(Data.Dataset):

    def ToTensor(self, x, y):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        y_tensor = torch.tensor(y).type(torch.FloatTensor)
        y_tensor = torch.unsqueeze(y_tensor,dim=0)
        return x_tensor, y_tensor   
        
    def __init__(self, dataroot):
        with open(dataroot,'rb') as handle:
            self.data = pickle.load(handle)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        global nch
        pack = self.data[idx]
        x = util.ImageRescale(pack[:nch,:,:],[0,255])
        y = util.ImageRescale(pack[-1,:,:],[0,1])
        x_tensor, y_tensor = self.ToTensor(x,y)
        return x_tensor, y_tensor
    
sf_loader = Data.DataLoader(dataset=MyDataset(dataroot),
                              batch_size=1, shuffle=True)

#%%
for step,[x,y] in enumerate(sf_loader):
    pass
print(x.size())
print(y.size())

#%%
Nsf = Arch.MS_UNet(gpu,nch).to(device)
Nsf.apply(Arch.weight_init)

L1_loss = nn.L1Loss()

beta1 = 0.5
lr = 1e-4
optimizer = optim.Adam(Nsf.parameters(),lr=lr,betas=(beta1,0.999))
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

print('Training start...')

t1 = time.time()
for epoch in range(nepoch):
    for step,[x,y] in enumerate(sf_loader):
        
        Nsf.train()
        
        train_x = Variable(x).to(device)        
        train_y = Variable(y).to(device)
        pred = Nsf(train_x)
                
        loss = L1_loss(pred,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss:%.4f'
                  %(epoch, nepoch, step, len(sf_loader),loss))
            
        if step % 500 == 0:
            with torch.no_grad():
                
                im_grand = np.zeros([512,1000],dtype=np.float32)
                
                denoise = pred.detach().cpu().numpy()
                avg = train_y.detach().cpu().numpy()
                
                im_grand[:,:500] = util.ImageRescale(denoise[0,0,:,:500],[0,255])
                im_grand[:,500:1000] = util.ImageRescale(avg[0,0,:,:500],[0,255])
                
                plt.figure(figsize=(12,6))
                plt.axis('off')
                plt.imshow(im_grand,cmap='gray')
                plt.show()
                
    scheduler.step()
    
#%%
name = 'HN2SF_nch=3.pt'
torch.save(Nsf.state_dict(),modelroot+name)
