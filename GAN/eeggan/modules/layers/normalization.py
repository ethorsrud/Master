# coding=utf-8
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

class PixelNorm(nn.Module):
	"""
	References
	----------
	Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
	Progressive Growing of GANs for Improved Quality, Stability, and Variation.
	Retrieved from http://arxiv.org/abs/1710.10196
	"""
	def forward(self,input,eps=1e-8):
		tmp = torch.sqrt(torch.pow(input,2).mean(dim=1,keepdim=True)+eps)
		input = input/tmp
		return input

class LayerNorm(Module):
	"""
	References
	----------
	Ba, J. L., Kiros, J. R., & Hinton, G. E. (n.d.). Layer Normalization.
	Retrieved from https://arxiv.org/pdf/1607.06450.pdf
	"""
	def __init__(self,num_features,n_dim,eps=1e-5,affine=True):
		assert(n_dim>1)

		super(LayerNorm, self).__init__()
		self.num_features = num_features
		self.n_dim = n_dim

		tmp_ones = [1]*(n_dim-2)
		self.affine = affine
		self.eps = eps
		if self.affine:
			self.weight = Parameter(torch.Tensor(1,num_features,*tmp_ones))
			self.weight.data.fill_(1.)
			self.bias = Parameter(torch.Tensor(1,num_features,*tmp_ones))
			self.bias.data.fill_(0.)
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		if self.affine:
			self.weight.data.uniform_()
			self.bias.data.zero_()

	def forward(self, input):
		orig_size = input.size()
		b = orig_size[0]
		tmp_dims = range(self.n_dim)

		trash_mean = torch.zeros(b)
		trash_var = torch.ones(b)
		if input.is_cuda:
			trash_mean = trash_mean.cuda()
			trash_var = trash_var.cuda()

		input_reshaped = input.contiguous().permute(1,0,*tmp_dims[2:]).contiguous()

		out = F.batch_norm(
			input_reshaped, trash_mean, trash_var, None, None,
			True, 0., self.eps).permute(1,0,*tmp_dims[2:]).contiguous()

		if self.affine:
			weight = self.weight
			bias = self.bias
			out = weight*out+bias

		return out
