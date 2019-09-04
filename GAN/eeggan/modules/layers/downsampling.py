# coding=utf-8
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Downsample1d(nn.Module):
	"""
	1d downsampling by only taking every n-th entry

	Parameters
	----------
	divisor : int
		Downscaling factor
	"""
	def __init__(self,divisor):
		super(Downsample1d,self).__init__()
		self.divisor = divisor

	def forward(self,input):
		input = input.contiguous().view(input.size(0),input.size(1),
							input.size(2)/self.divisor,self.divisor)
		input = input[:,:,:,0]
		return input

class Downsample2d(nn.Module):
	"""
	2d downsampling by only taking every n-th entry

	Parameters
	----------
	divisor : (int,int)
		Downscaling factors
	"""
	def __init__(self,divisor):
		super(Downsample2d,self).__init__()
		self.divisor = divisor

	def forward(self,input):
		input = input.contiguous().view(input.size(0),input.size(1),
							input.size(2)/self.divisor[0],self.divisor[0],
							input.size(3)/self.divisor[1],self.divisor[1])
		input = input[:,:,:,0,:,0]
		return input
