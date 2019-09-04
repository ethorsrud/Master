# coding=utf-8
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch
import numpy as np

class SpectralNorm(object):
	"""
	Implemented for PyTorch using WeightNorm implementation
	https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html

	References
	----------
	Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018).
	Spectral Normalization for Generative Adversarial Networks.
	Retrieved from http://arxiv.org/abs/1802.05957
	"""
	def __init__(self, name):
		self.name = name

	def compute_weight(self, module):
		weight = getattr(module, self.name)
		u = getattr(module, self.name + '_u')

		weight_size = list(weight.size())
		weight_tmp = weight.data.view(weight_size[0],-1)
		v = weight_tmp.t().matmul(u)
		v = v/v.norm()
		u = weight_tmp.matmul(v)
		u = u/u.norm()
		o = u.t().matmul(weight_tmp.matmul(v))
		weight_tmp = weight_tmp/o
		weight.data = weight_tmp.view(*weight_size)

		setattr(module, self.name + '_u', u)
		setattr(module, self.name, weight)

	@staticmethod
	def apply(module, name):
		fn = SpectralNorm(name)

		weight = getattr(module, name)
		u = torch.Tensor(weight.size(0),1)
		u.normal_()

		module.register_buffer(name + '_u', u)
		module.register_forward_pre_hook(fn)

		return fn

	def remove(self, module):
		del module._buffers[name + '_u']

	def __call__(self, module, input):
		self.compute_weight(module)


def spectral_norm(module, name='weight', dim=0):
	SpectralNorm.apply(module, name)
	return module

def remove_spectral_norm(module, name='weight'):
	for k, hook in module._forward_pre_hooks.items():
		if isinstance(hook, SpectralNorm) and hook.name == name:
			hook.remove(module)
			del module._forward_pre_hooks[k]
			return module

	raise ValueError("weight_norm of '{}' not found in {}"
					 .format(name, module))
