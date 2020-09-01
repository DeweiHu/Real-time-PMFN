# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:00:04 2020

@author: hudew
"""

import torch
import torch.nn as nn

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def down_block(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )
    
def up_block(in_channels,out_channels):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )

def Standard_block(in_channels,out_channels):
     return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1)
            )

def double_path(in_channels,d):
    return nn.Sequential(
           nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3,
                     stride=1, padding=d, dilation=d),
           nn.ELU(),
           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                     stride=1, padding=d, dilation=d),
           nn.BatchNorm2d(64),
           nn.ELU()
           )

def single_path(in_channels,d):
    return nn.Sequential(
           nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3,
                     stride=1, padding=d, dilation=d),
           nn.ELU()
           )
    
def conv(in_channels,out_channels):
    return nn.Sequential(
           nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=1),
           nn.ELU()
           )
    
    
class MS_UNet(nn.Module):
    def __init__(self,gpu,in_channel):
        super(MS_UNet,self).__init__()
        self.gpu = gpu
        
        self.down = down_block(64,64)
        self.up = up_block(64,64)
        
        self.Standard_1 = Standard_block(in_channel,64)
        
        self.Res1_double = double_path(64,2)
        self.Res1_single = single_path(64,2)
        
        self.Res2_double = double_path(64,4)
        self.Res2_single = single_path(64,4)

        self.Standard_2 = Standard_block(64,64)
        
        self.Res3_double = double_path(128,4)
        self.Res3_single = single_path(128,4)
        
        self.Standard_3 = Standard_block(128,64)
        
        self.conv = conv(64,64)
        self.convopt = conv(256,1)
        
    def forward(self,x):
        # Downwards
        layer_1 = self.Standard_1(x)           #[64,1024,512]
        x = self.down(layer_1)                 #[64,512,256]
        
        layer_2 = torch.add(self.Res1_double(x),self.Res1_single(x))  #[64,512,256]
        x = self.down(layer_2)                                        #[64,256,128]

        layer_3 = torch.add(self.Res2_double(x),self.Res2_single(x))  #[64,256,128]
        x = self.down(layer_3)                                        #[64,128,64]
        
        x = self.Standard_2(x)
        
        # Upwards
        x = self.up(x)                                          #[64,256,128]
        x = torch.cat([x,layer_3],dim=1)                        #[128,256,128]
        x = torch.add(self.Res3_double(x),self.Res3_single(x))  #[64,256,128]

        x = self.up(x)                                          #[64,512,256]
        x = torch.cat([x,layer_2],dim=1)                        #[128,512,256]
        x = torch.add(self.Res3_double(x),self.Res3_single(x))  #[64,512,256]
        
        x = self.up(x)                         #[64,1024,512]
        x = torch.cat([x,layer_1],dim=1)       #[128,1024,512]
        x = self.Standard_3(x)                 #[64,1024,512]
        
        # Middle feedbacks
        l1 = self.conv(layer_1)
        l2 = self.up(self.conv(layer_2))
        l3 = self.up(self.up(self.conv(layer_3)))
        x = torch.cat([x,l1],dim=1)
        x = torch.cat([x,l2],dim=1)
        x = torch.cat([x,l3],dim=1)
        
        output = self.convopt(x)
        return output