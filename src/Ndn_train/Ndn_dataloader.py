import sys
sys.path.insert(0,'E:\\real-time-PMFN\\')
import util
import MotionCorrection as MC

import os,pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage import io

import cv2
import NetworkArch
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

'''
X:
    Channel 1: original noisy B-scan
    Channel 2: output of Nsf
    Channel 3: gradient map

Y:
    y1: n-frame average
    y2: SF(n-frame average)
'''

# radius for self-fusion neighborhood
global radius
radius = 3

def Sobel(img, kernel_size):
    sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=kernel_size)
    gradient = np.sqrt(np.square(sobelx)+np.square(sobely))
    gradient *= 255.0/gradient.max()
    return np.float32(gradient)

class Nsf_test_dataset(Data.Dataset):
    
    def ToTensor(self, x):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        return x_tensor
    
    def __init__(self,root):
        self.vx = util.nii_loader(root)
        n,h,w = self.vx.shape
        
        self.data = []
        for i in range(radius,n-radius+1):
            x = np.zeros([3,512,512],dtype=np.float32)
            
            im_fix = np.ascontiguousarray(np.float32(self.vx[i,:,:]))
            
            mov_pre = np.ascontiguousarray(np.float32(self.vx[i-1,:,:]))
            mov_post = np.ascontiguousarray(np.float32(self.vx[i+1,:,:]))
            
            x[0,:,:w] = MC.MotionCorrect(im_fix,mov_pre)
            x[2,:,:w] = MC.MotionCorrect(im_fix,mov_post)
            x[1,:,:w] = im_fix
            
            self.data.append(x)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        x = self.data[idx]        
        x_tensor = self.ToTensor(x)
        return x_tensor

# Load model
gpu = 1
nch = 3
modelroot = 'E:\\Model\\'
device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")

Nsf = NetworkArch.MS_UNet(gpu,nch).to(device)
Nsf.load_state_dict(torch.load(modelroot+'HN2SF_nch=3.pt'))

#%% main
        
if __name__=='__main__':

    dataroot = 'E:\\real-time-PMFN\\data\\'
    
    hn_list = []
    ln_list = []
    sf_list = []
    
    for file in os.listdir(dataroot):
        if file.startswith('HN_ONH'):
            hn_list.append(file)
        elif file.startswith('LN_ONH'):
            ln_list.append(file)
        elif file.startswith('SF_ONH'):
            sf_list.append(file)
    
    hn_list.sort()
    ln_list.sort()
    sf_list.sort()
    
    pair_data = ()
    for i in range(len(hn_list)):
        
        ln = util.nii_loader(dataroot+ln_list[i])
        sf_ln = util.nii_loader(dataroot+sf_list[i])
        
        Nsf_test_loader = Data.DataLoader(dataset=Nsf_test_dataset(dataroot+hn_list[i]))
        print('dataloader {} created'.format(hn_list[i]))
        
        for step,x in enumerate(Nsf_test_loader):
            with torch.no_grad():
                x = Variable(x).to(device)
                pred = Nsf(x).detach.cpu().numpy()
                
                # high noise bscan -- Nsf(x) -- sobel
                x_stack = np.zeros([3,512,512],dtype=np.float32)
                
                bscans = x.detach().cpu().numpy()
                x_stack[0,:,:] = bscans[0,1,:,:] 
                x_stack[1,:,:500] = util.ImageRescale(pred[0,0,:,:500],[0,255])
                x_stack[2,:,:500] = Sobel(x_stack[1,:,:500],3)
                
                y1 = np.zeros([512,512],dtype=np.float32)
                y1[:,:500] = ln[step,:,:500]
                
                y2 = np.zeros([512,512],dtype=np.float32)
                y2[:,:500] = sf_ln[step,:,:500]
                
                pair_data = pair_data+((x_stack,y1,y2),)
                
            if step == 200:
                plt.figure(figsize=(10,5))
                plt.imshow(np.concatenate((x_stack[0,:,:],x_stack[1,:,:]),axis=1))
                plt.show()
                
                plt.figure(figsize=(10,5))
                plt.imshow(np.concatenate((y1[0,:,:],y2[1,:,:]),axis=1))
                plt.show()

#%%
    with open(dataroot+'Ndn_train_data.pickle','wb') as handle:
        pickle.dump(pair_data,handle)

