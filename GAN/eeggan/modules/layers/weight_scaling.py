# coding=utf-8
import torch
from torch import nn
import numpy as np
from eeggan.modules.layers.multiconv import MultiConv1d
from torch.autograd import Variable
import torch.nn.functional as F

class WeightScale(object):
	"""
	Implemented for PyTorch using WeightNorm implementation
	https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html

	References
	----------
	Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
	Progressive Growing of GANs for Improved Quality, Stability,
	and Variation. Retrieved from http://arxiv.org/abs/1710.10196
	"""
	def __init__(self, name):
		self.name = name

	def compute_weight(self, module):
		w = getattr(module, self.name + '_unscaled')
		c = getattr(module, self.name + '_c')
		tmp = c*w
		return tmp

	@staticmethod
	def apply(module, name, gain):
		fn = WeightScale(name)
		weight = getattr(module, name)
		# remove w from parameter list
		del module._parameters[name]

		#Constant from He et al. 2015
		c = gain/np.sqrt(np.prod(list(weight.size())[1:]))
		setattr(module, name + '_c', float(c))
		module.register_parameter(name + '_unscaled', nn.Parameter(weight.data))
		setattr(module, name, fn.compute_weight(module))
		# recompute weight before every forward()
		module.register_forward_pre_hook(fn)
		return fn

	def remove(self, module):
		weight = self.compute_weight(module)
		delattr(module, self.name)
		del module._parameters[self.name + '_unscaled']
		del module._parameters[self.name + '_c']
		module.register_parameter(self.name, Parameter(weight.data))

	def __call__(self, module, inputs):
		setattr(module, self.name, self.compute_weight(module))

def weight_scale(module, gain=np.sqrt(2), name='weight'):
	"""
	Applies equalized learning rate to weights

	Parameters
	----------
	module : module
		Module scaling should be applied to (Conv/Linear)
	gain : float
		Gain of following activation layer
		See torch.nn.init.calculate_gain
	"""
	if isinstance(module,MultiConv1d):
		for i in range(len(module.convs)):
			WeightScale.apply(module.convs[i], name, gain)
	else:
		WeightScale.apply(module, name, gain)
	return module

def remove_weight_scale(module, name='weight'):
	for k, hook in module._forward_pre_hooks.items():
		if isinstance(hook, WeightScale) and hook.name == name:
			hook.remove(module)
			del module._forward_pre_hooks[k]
			return module

	raise ValueError("weight_scale of '{}' not found in {}"
					 .format(name, module))
