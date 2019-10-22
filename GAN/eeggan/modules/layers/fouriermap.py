# coding=utf-8
import torch
from torch import nn
import numpy as np

class FFTMap1d(nn.Module):
    def __init__(self):
        super(FFTMap1d,self).__init__()

    def forward(self,input):
        fft = torch.rfft(input,1,normalized=True)
        fft = torch.sqrt(fft[:,:,1:,0]**2+fft[:,:,1:,1]**2)
        return input