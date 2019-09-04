# coding=utf-8
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Reshape(nn.Module):
	"""
	Reshape tensor into new shape

	Parameters
	----------
	shape : list
		New shape
		Follows numpy reshaping
	"""
	def __init__(self,shape):
		super(Reshape,self).__init__()
		self.shape = shape

	def forward(self,input):
		shape = list(self.shape)
		for i in range(len(shape)):
			if type(shape[i]) is list or type(shape[i]) is tuple:
				assert len(shape[i])==1
				shape[i] = input.size(shape[i][0])
		return input.view(shape)

class PixelShuffle1d(nn.Module):
	"""
	1d pixel shuffling
	Shuffles filter dimension into trailing dimension

	Parameters
	----------
	scale_kernel : int
		Factor of how many filters are shuffled

	References
	----------
	Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R.,
	… Wang, Z. (2016).
	Real-Time Single Image and Video Super-Resolution Using an Efficient
	Sub-Pixel Convolutional Neural Network.
	Retrieved from http://arxiv.org/abs/1609.05158
	"""
	def __init__(self,scale_kernel):
		super(PixelShuffle1d, self).__init__()
		self.scale_kernel = scale_kernel

	def forward(self, input):
		batch_size, channels, in_height = input.size()
		channels /= self.scale_kernel[0]

		out_height = in_height * self.scale_kernel[0]

		input_view = input.contiguous().view(
			batch_size, channels, self.scale_kernel[0],in_height)

		shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
		return shuffle_out.view(batch_size, channels, out_height)

class PixelShuffle2d(nn.Module):
	"""
	2d pixel shuffling
	Shuffles filter dimension into trailing dimensions

	Parameters
	----------
	scale_kernel : (int,int)
		Factors of how many filters are shuffled

	References
	----------
	Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R.,
	… Wang, Z. (2016).
	Real-Time Single Image and Video Super-Resolution Using an Efficient
	Sub-Pixel Convolutional Neural Network.
	Retrieved from http://arxiv.org/abs/1609.05158
	"""
	def __init__(self,scale_kernel):
		super(PixelShuffle2d, self).__init__()
		self.scale_kernel = scale_kernel

	def forward(self, input):
		batch_size, channels, in_height, in_width = input.size()
		channels /= self.scale_kernel[0]*self.scale_kernel[1]

		out_height = in_height * self.scale_kernel[0]
		out_width = in_width * self.scale_kernel[1]

		input_view = input.contiguous().view(
			int(batch_size), int(channels), int(self.scale_kernel[0]), int(self.scale_kernel[1]),
			int(in_height), int(in_width))

		shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
		return shuffle_out.view(int(batch_size), int(channels), int(out_height), int(out_width))
