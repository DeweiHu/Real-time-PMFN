# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:54:15 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import util
import MotionCorrection as MC

import os, time, pickle
import numpy as np
import matplotlib.pyplot as plt

def Display(x,y):
    plt.figure(figsize=(12,6))
    plt.axis('off')
    plt.imshow(np.concatenate((x,y),axis=1),cmap='gray')
    plt.show()

def GetPair(HN, SF, opt):
    global nch
    nBscan,H,W = HN.shape
    radius = int((nch-1)/2)
    for i in range(nBscan):
        if i >= radius and i < nBscan-radius:
            x = np.zeros([nch,512,512],dtype=np.float32)
            y = SF[i,:,:]
            
            # eliminate the zero pad for im_fix, mov_pre and mov_post 
            im_fix = np.ascontiguousarray(np.float32(HN[i,:,:500]))
            x[radius,:,:500] = im_fix          
            
            for j in range(radius):
                dist = j+1
                mov_pre = np.ascontiguousarray(np.float32(HN[i-dist,:,:500]))
                reg_pre = MC.MotionCorrect(im_fix,mov_pre)
                x[radius-dist,:,:500] = reg_pre 
                
                mov_post = np.ascontiguousarray(np.float32(HN[i+dist,:,:500]))
                reg_post = MC.MotionCorrect(im_fix,mov_post)
                x[radius+dist,:,:500] = reg_post    
            
            opt.append(np.concatenate((x,np.expand_dims(y,axis=0)),axis=0))        
        
            # display an example
            if i == 200:
               Display(x[radius,:,:],y) 
    return opt

#%%
dataroot = 'E:\\HumanData\\'
HN_list = []
SF_list = []

for file in os.listdir(dataroot):
    if file.startswith('HN'):
        HN_list.append(file)
    elif file.startswith('SF'):
        SF_list.append(file)
HN_list.sort()
SF_list.sort()

# use only 2 adjacent neighbors
global nch
nch = 3
train_data = []

for i in range(len(HN_list)):
    HN = util.nii_loader(dataroot+HN_list[i])
    HN = HN[3:-3,:,:] # hard coded, radius of self-fusion used
    SF = util.nii_loader(dataroot+SF_list[i])
    
    print('volume {} pairing...'.format(i+1))
    train_data = GetPair(HN, SF, train_data)

with open(dataroot+'PM_traindata.pickle','wb') as handle:
    pickle.dump(train_data,handle)
    
print('done.')
    
    

