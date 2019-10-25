# coding=utf-8
import torch
from torch import nn
import numpy as np

class FFTMap1d(nn.Module):
    def __init__(self):
        super(FFTMap1d,self).__init__()

    def forward(self,input):
        fft = torch.transpose(torch.rfft(torch.transpose(input,2,3),1,normalized=True),2,3)
        fft = torch.sqrt(fft[:,:,1:,:,0]**2+fft[:,:,1:,:,1]**2)
        upsampler = torch.nn.Upsample(scale_factor=2,mode='linear')
        fft = torch.transpose(upsampler(torch.transpose(fft[:,0,:,:],1,2)),1,2)
        fft = torch.unsqueeze(fft,1)
        centered = fft-torch.mean(fft,dim=0)
        fft = centered/torch.std(fft,dim=0)
        fft = fft/torch.max(fft)
        fft = torch.cat((input,fft),dim=3)
        return fft
