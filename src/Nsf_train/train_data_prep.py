import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import util
import MotionCorrection as MC

import os, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

'''
Re_Arrange reshape the volume from [nFrame*nBscan,H,W] -> [nFrame,nBscan,H,W]
'''
def Re_Arrange(volume):
    global nFrame
    n,H,W = volume.shape
    opt = np.zeros([nFrame,int(n/nFrame),H,W],dtype=np.float32)
    for i in range(n):
        idx = i % nFrame
        opt[idx,int(i/nFrame),:,:] = volume[i,:,:]
    return opt

'''
FrameAver do rigid registration on repeated frames then take the average
idx: which frame is fixed
 input shape: [nFrame,nBscan,H,W]
output shape: [nBscan,H,W]
'''
def FrameAver(volume,idx):
    nFrame,nBscan,H,W = volume.shape
    opt = np.zeros([nFrame,nBscan,H,W],dtype=np.float32)
    # iter over Bscan
    for i in range(nBscan):
        im_fix = np.ascontiguousarray(np.float32(volume[idx,i,:,:])) 
        # iter over frames
        for j in range(nFrame):
            im_mov = np.ascontiguousarray(np.float32(volume[j,i,:,:]))
            opt[j,i,:,:] = MC.MotionCorrect(im_fix,im_mov)
    return np.mean(opt,axis=0)

'''
BscanRegist do the rigid registration to neighboring B-scans and stack them
'''
def BscanRegist(LN, radius, verbose):
    [nBscan,H,W] = LN.shape
    nch = radius*2+1
    opt = []
    for i in range(nBscan):
        # get neighboring slices within radius
        if i >= radius and i < nBscan-radius:
            # zero-pad the W
            x = np.zeros([nch,512,512],dtype=np.float32)           
            im_fix = np.ascontiguousarray(np.float32(LN[i,:,:]))
            x[radius,:,:500] = im_fix
            
            for j in range(radius):
                dist = j+1
                mov_pre = np.ascontiguousarray(np.float32(LN[i-dist,:,:]))
                reg_pre = MC.MotionCorrect(im_fix,mov_pre)
                x[radius-dist,:,:500] = reg_pre[:512,:] 
                
                mov_post = np.ascontiguousarray(np.float32(LN[i+dist,:,:]))
                reg_post = MC.MotionCorrect(im_fix,mov_post)
                x[radius+dist,:,:500] = reg_post[:512,:]               
            opt.append(x)
            
        # display a sample
        if verbose == True and i == 200:
            plt.figure(figsize=(12,4))
            plt.axis('off'),plt.title('Regist Result',fontsize=15)
            plt.imshow(np.concatenate((x[0,:,:],x[radius+1,:,:],x[-1,:,:]),axis=1),cmap='gray')
            plt.show()   
    return opt

#%% Data re-arrange and LN creation
dataroot = 'E:\\human tiff\\'
vlist = []

for file in os.listdir(dataroot):
    if file.startswith('Retina2_ONH'):
        vlist.append(file)

global nFrame, fixFrame, sf_r
nFrame = 5
fixFrame = 0
sf_r = 3

t1 = time.time()

for i in range(len(vlist)):
    print('Creating low noise volume....')
    name = vlist[i]
    # rescaling -> crop-out the massive back ground -> reshape
    HN = util.ImageRescale(io.imread(dataroot+name),[0,255])
    HN = Re_Arrange(HN[:,:512,:])
    # Frame-average 
    LN = FrameAver(HN,fixFrame)
    
    util.nii_saver(HN[fixFrame,:,:,:],'E:\\HumanData\\','HN_{}.nii.gz'.format(name[8:-4]))
    util.nii_saver(LN,'E:\\HumanData\\','LN_{}.nii.gz'.format(name[8:-4]))
    
    # display a sample
    plt.figure(figsize=(10,5))
    plt.axis('off'),plt.title('Volume #{}:'.format(i+1)+name,fontsize=15)
    plt.imshow(np.concatenate((HN[0,100,:,:],LN[100,:,:]),axis=1),cmap='gray')
    plt.show()
    
    # Bscan registration for self-fusion
    print('Creating self-fusion data....')
    sf_data = BscanRegist(LN, sf_r, True)
    with open('E:\\HumanData\\'+name[8:-4]+'.pickle','wb') as handle:
        pickle.dump(sf_data,handle)
    del sf_data, HN, LN
    
t2 = time.time()
print('LN volume & SF(LN) created. Time used:{} min.'.format((t2-t1)/60))
