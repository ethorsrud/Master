# coding=utf-8
from torch.autograd import Variable
from torch.nn import Module

def cuda_check(module_list):
	"""
	Checks if any module or variable in a list has cuda() true and if so
	moves complete list to cuda

	Parameters
	----------
	module_list : list
		List of modules/variables

	Returns
	-------
	module_list_new : list
		Modules from module_list all moved to the same device
	"""
	cuda = False
	for mod in module_list:
		if isinstance(mod,Variable): cuda = mod.is_cuda
		elif isinstance(mod,Module): cuda = next(mod.parameters()).is_cuda

		if cuda:
			break
	if not cuda:
		return module_list

	module_list_new = []
	for mod in module_list:
		module_list_new.append(mod.cuda())
	return module_list_new


def change_learning_rate(optimizer,lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def weight_filler(m):
	classname = m.__class__.__name__
	if classname.find('MultiConv') != -1:
		for conv in m.convs:
			conv.weight.data.normal_(0.0, 1.)
			if conv.bias is not None:
				conv.bias.data.fill_(0.)
	elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 1.) # From progressive GAN paper
		if m.bias is not None:
			m.bias.data.fill_(0.)
	elif classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0.)
