# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:37:56 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import util
import MotionCorrection as MC

import os
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
PickFrame separate the stacked repeated frames into single frame volume
FrameNum : number of repeated frame
idx : which frame to pick
'''
def PickFrame(volume,FrameNum,idx):
    dim = volume.shape
    opt = np.zeros([int(dim[0]/FrameNum),dim[1],dim[2]],dtype=np.float32)
    for i in range(dim[0]):
        if i % FrameNum == idx:
            opt[int(i/FrameNum),:,:] = volume[i,:,:]
    return opt

'''
BscanRegist do the rigid registration to neighboring B-scans and stack them
'''
def BscanRegist(Volume_noi,nch):
    [nd,nr,nc] = Volume_noi.shape
    radius = int((nch-1)/2)
    opt = []
    for i in range(nd):
        if i >= radius and i < nd-radius:
            x = np.zeros([nch,512,512],dtype=np.float32)           
            fix = np.ascontiguousarray(np.float32(Volume_noi[i,:,:]))
            x[radius,:,:500] = fix[:512,:]          
            for j in range(radius):
                dist = j+1
                mov_pre = np.ascontiguousarray(np.float32(Volume_noi[i-dist,:,:]))
                reg_pre = MC.MotionCorrect(fix,mov_pre)
                x[radius-dist,:,:500] = reg_pre[:512,:] 
                
                mov_post = np.ascontiguousarray(np.float32(Volume_noi[i+dist,:,:]))
                reg_post = MC.MotionCorrect(fix,mov_post)
                x[radius+dist,:,:500] = reg_post[:512,:]               
            opt.append(x)          
        if i % 50 == 0:
            print('[%d/%d] complete'%(i,nd))    
    return opt

'''
Sobel edge detector, the output is the gradient magnitude map
'''
def Sobel(img, kernel_size):
    sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=kernel_size)
    gradient = np.sqrt(np.square(sobelx)+np.square(sobely))
    gradient *= 255.0/gradient.max()
    return np.float32(gradient)

'''
DataLoader for networks
'''
class MyDataset(Data.Dataset):

    def ToTensor(self, image):
        x_tensor = torch.tensor(image).type(torch.FloatTensor)
        return x_tensor   
        
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        image = self.data[idx]
        x_tensor = self.ToTensor(image)
        return x_tensor
'''
Main function of PMFN
raw_Volume: noisy .tiff volume 
FrameNum  : number of repeated frame
nch       : number of input channel to synthesize pseudo-modality
Nsf       : Network 1
Ndn       : Network 2   
'''
def PMFN_main(raw_Volume,FrameNum,nch,Nsf,Ndn,sf_display,dn_display):
    # Pre-define the output form
    depth = 500-nch+1
    opt = np.zeros([FrameNum,depth,512,500],dtype=np.float32)
    
    for idx in range(FrameNum): 
        
        # [1] Frame separation and croping
        volume = PickFrame(raw,FrameNum,idx)
        volume = volume[:,150:662,:]
        print('Frame {} separated.'.format(idx))

        # [2] Bscan Registration
        print('Registering Bscans...')
        test_x = BscanRegist(volume,nch)
        
        # [3] noise to self-fusion-y
        print('Predicting self-fusion-y...')
        denoise_x = []
        sf_loader = Data.DataLoader(dataset=MyDataset(test_x),
                              batch_size=1, shuffle=False)

        for step,x in enumerate(sf_loader):
            with torch.no_grad():
                x = Variable(x).to(device)
                pred = Nsf(x).detach().cpu().numpy()
                
                x_opt = np.zeros([3,512,512],dtype=np.float32)
                
                bscans = x.detach().cpu().numpy()
                x_opt[0,:,:] = bscans[0,3,:,:]
                
                pred = util.ImageRescale(pred[0,0,:,:500],[0,255])
                gradient = Sobel(pred,3)
                
                x_opt[1,:,:500] = gradient
                x_opt[2,:,:500] = pred
                denoise_x.append(x_opt)
                
                if step % 20 == 0 and sf_display == True: 
                    plt.figure(figsize=(18,6))
                    plt.title('slc:{}'.format(step))
                    plt.axis('off')
                    plt.imshow(np.concatenate([x_opt[0,:,:500],pred,gradient],axis=1),cmap='gray')
                    plt.show()
        del sf_loader
        
        # [4] multi-modal input to denoised
        print('De-speckling...')
        dn_loader = Data.DataLoader(dataset=MyDataset(denoise_x),
                              batch_size=1, shuffle=False)

        for step,x in enumerate(dn_loader):
            with torch.no_grad():
                
                x = Variable(x).to(device)
                noi = x[0,0,:,:500].detach().cpu().numpy()
                pred = Ndn(x).detach().cpu().numpy()
                opt[idx,step,:,:] = util.ImageRescale(pred[0,0,:,:500],[0,255])
                
                if step % 20 == 0 and dn_display == True: 
                    plt.figure(figsize=(18,6))
                    plt.title('slc:{}'.format(step))
                    plt.axis('off')
                    plt.imshow(np.concatenate([noi,opt[idx,step,:,:]],axis=1),cmap='gray')
                    plt.show()
        del dn_loader
        print('------------------------------------------------')
        
    return opt

'''
Main function of MS-UNet
'''
def MSUN_main(raw_Volume,FrameNum,Ndn,dn_display):
    # Pre-define the output form
    depth = 500-nch+1
    opt = np.zeros([FrameNum,depth,512,500],dtype=np.float32)
    
    for idx in range(FrameNum): 
        
        # [1] Frame separation and croping
        volume = PickFrame(raw,FrameNum,idx)
        volume = volume[:,150:662,:]
        print('Frame {} separated.'.format(idx))
        
        for slc in range(depth):
            noi = volume[slc,:,:500]
            x = torch.tensor(volume[slc,:,:]).type(torch.FloatTensor)
            x = Variable(x).to(device)
            pred = Ndn(x).detach().cpu().numpy()
            opt[idx,slc,:,:] = util.ImageRescale(pred[0,0,:,:500],[0,255])
            
            if slc % 50 == 0 and dn_display == True: 
                plt.figure(figsize=(18,6))
                plt.title('slc:{}'.format(slc))
                plt.axis('off')
                plt.imshow(np.concatenate([noi,opt[idx,slc,:,:]],axis=1),cmap='gray')
                plt.show()
            
    return opt

#%% load models
gpu = 1
nch_1 = 7
nch_2 = 3
modelroot = 'E:\\Model\\'
device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")

Nsf = NetworkArch.MS_UNet(gpu,nch_1).to(device)
Nsf.load_state_dict(torch.load(modelroot+'noi2sf_MSUNet.pt'))

Ndn = NetworkArch.MS_UNet(gpu,nch_2).to(device)
Ndn.load_state_dict(torch.load(modelroot+'1_12.pt'))

#%% load the raw data

if __name__=='__main__':
    
    root = 'E:\\human\\' 
    volumelist = []
    
    for file in os.listdir(root):
        if file.startswith('Retina2_Fovea') and file.endswith('.tif'):
            volumelist.append(file)
    volumelist.sort()
    
    volume = volumelist[0]
    raw = util.ImageRescale(io.imread(root+volume),[0,255])
    
    FrameNum = 5
    nch = 7
    
    t1 = time.time() 
    V = PMFN_main(raw,FrameNum,nch,Nsf,Ndn,False,False)
    t2 = time.time()
    print('Denoised finish, time used: {} min'.format((t2-t1)/60))
    
    util.nii_saver(V,'E:\\Retina2_Fovea\\101_1\\','MSUN_101.nii.gz')

