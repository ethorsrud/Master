# coding=utf-8
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

class add_random_layer(nn.Module):
    """
    adding a random numbers from normal[0,1]
    """
    def forward(self,input):
        print(input.shape)
        quit()
        return input