#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:00:33 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/20-summer/src/')
import util

import time, pickle, os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

def Display(x,y):
    plt.figure(figsize=(12,6))
    plt.axis('off')
    plt.imshow(np.concatenate((x,y),axis=1),cmap='gray')
    plt.show()

dataroot = '/home/hud4/Desktop/2020/Data/'
Dir = '/home/hud4/Desktop/regi_result/'
radius = 3

# clean the temp dir in advance
for tif in os.listdir(Dir):
    os.remove(Dir+tif)
                
# Load data from pickle
idx = 1

t1 = time.time()
for file in os.listdir(dataroot):
    if file.endswith('.pickle'):
        with open(dataroot+file,'rb') as handle:
            data = pickle.load(handle)
        
        print('volume {} self-fusing...'.format(idx))
        _,H,W = data[0].shape
        v_sf = np.zeros([len(data),H,W],dtype=np.float32)
        
        for i in range(len(data)):
            # define fix image
            pack = data[i]
            x = pack[radius,:,:]
            im_fix = Image.fromarray(x)
            im_fix.save(Dir+'fix_img.tif')
            # define atlases
            for j in range(radius*2+1):
                im_mov = Image.fromarray(pack[j,:,:])
                im_mov.save(Dir+'atlas{}.tif'.format(j))
            
            # call self-fusion function
            subprocess.call('/home/hud4/Desktop/20-summer/src/self_fusion.sh')      
            v_sf[i,:,:] = io.imread(Dir+'synthResult.tif')
            
            # display a sample
            if i == 200:
                Display(x,v_sf[i,:,:])
            
            # clean up the temp directory
            for tif in os.listdir(Dir):
                os.remove(Dir+tif)
        
        util.nii_saver(v_sf,'/home/hud4/Desktop/2020/',file[:-7]+'.nii.gz')
        idx += 1
        
t2 = time.time()
print('Self-fusion done. Time: {} min'.format((t2-t1)/60))