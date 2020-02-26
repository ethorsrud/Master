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
        normal = torch.distributions.normal.Normal(0,1, validate_args=None)
        z_vars = normal.sample(sample_shape=(input.shape))
        if input.is_cuda:
            z_vars = z_vars.cuda()
        return torch.cat((input,z_vars),dim=1)