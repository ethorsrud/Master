# coding=utf-8
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

class upsample_layer(Module):
	def __init__(self,mode,scale_factor):
		super(upsample_layer,self).__init__()
		self.mode = mode
		self.scale_factor = scale_factor

	def forward(self,input):
		return F.interpolate(input,mode=self.mode,scale_factor=self.scale_factor,align_corners=False)

class Upscale1d(Module):
	"""
	1d Nearest-neighbor upsampling

	Parameters
	----------
	scale_factor : int
		Size increase
	"""
	def __init__(self,scale_factor):
		super(Upscale1d, self).__init__()
		self.scale_factor = scale_factor

	def forward(self,input):
		input_shape = input.size()
		tmp = input.view(input_shape[0],input_shape[1],
							input_shape[2],1)
		tmp = tmp.expand(input_shape[0],input_shape[1],
							input_shape[2],self.scale_factor)
		tmp = tmp.contiguous().view(input_shape[0],input_shape[1],
							input_shape[2]*self.scale_factor)
		return tmp

class Upscale2d(Module):
	"""
	2d Nearest-neighbor upsampling

	Parameters
	----------
	scale_factor : (int,int)
		Size increase for each dimension
	"""
	def __init__(self,scale_factor):
		super(Upscale2d, self).__init__()
		self.scale_factor = scale_factor

	def forward(self,input):
		input_shape = input.size()
		tmp = input.view(input_shape[0],input_shape[1],
							input_shape[2],1,input_shape[3],1)
		tmp = tmp.expand(input_shape[0],input_shape[1],
							input_shape[2],self.scale_factor[0],
							input_shape[3],self.scale_factor[1])
		tmp = tmp.contiguous().view(input_shape[0],input_shape[1],
							input_shape[2]*self.scale_factor[0],
							input_shape[3]*self.scale_factor[1])
		return tmp


def calc_kernel(t):
	t_arr = np.asarray([[1.,t,t**2,t**3]])
	w_arr = np.asarray([[0,2,0,0],
					   [-1,0,1,0],
					   [2,-5,4,-1],
					   [-1,3,-3,1]])
	return 1./2.*np.matmul(t_arr,w_arr)

class CubicUpsampling1d(nn.Module):
	"""
	1d Cubic upsampling

	Parameters
	----------
	scale_factor : int
		Size increase
	"""
	def __init__(self,scale_factor):
		super(CubicUpsampling1d,self).__init__()
		self.scale_factor = scale_factor
		dt = 1./scale_factor
		kernel = np.zeros((4,scale_factor)).astype(np.float32)
		for i in range(scale_factor-1):
			kernel[:,i] = calc_kernel((i+1)*dt)
		kernel[1,-1] = 1
		self.register_buffer('kernel', torch.from_numpy(kernel.flatten()[::-1].copy().reshape((1,1,-1))))

	def forward(self,input):
		old_size = input.size()
		input = F.pad(input,pad=(2,2),mode='replicate')
		#output = input
		weight = Variable(self.kernel.expand(old_size[1],-1,-1),requires_grad=False).contiguous()
		output = F.conv_transpose1d(input,weight,groups=old_size[1],stride=self.scale_factor)
		output = output[:,:,4*(self.scale_factor-1)+4:-(4*(self.scale_factor-1)+2)]
		return output

class CubicUpsampling2d(nn.Module):
	"""
	2d Cubic upsampling

	Parameters
	----------
	scale_factor : int
		Size increase
		Only works in one dimension, but can handle trailing singleton dimension
	"""
	def __init__(self,scale_factor):
		super(CubicUpsampling2d,self).__init__()
		self.scale_factor = scale_factor
		dt = 1./scale_factor
		kernel = np.zeros((4,scale_factor)).astype(np.float32)
		for i in range(scale_factor-1):
			kernel[:,i] = calc_kernel((i+1)*dt)
		kernel[1,-1] = 1
		self.register_buffer('kernel', torch.from_numpy(kernel.flatten()[::-1].copy().reshape((1,1,-1,1))))

	def forward(self,input):
		old_size = input.size()
		input = F.pad(input,pad=(0,0,2,2),mode='replicate')
		#output = input
		weight = Variable(self.kernel.expand(old_size[1],-1,-1,-1),requires_grad=False).contiguous()
		print(weight.size())
		output = F.conv_transpose2d(input,weight,groups=old_size[1],stride=(self.scale_factor,1))
		output = output[:,:,4*(self.scale_factor-1)+4:-(4*(self.scale_factor-1)+2),:]
		return output
