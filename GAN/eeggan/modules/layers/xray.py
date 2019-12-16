# coding=utf-8
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

class xrayscanner(nn.Module):
    """
    Just for debugging
    """
    def forward(self,input):
        input_for_investigation = input.data.cpu().numpy()
        print("IS IT FINITE?",np.all(np.isfinite(input_for_investigation)))
        print(input_for_investigation.shape)
        return input