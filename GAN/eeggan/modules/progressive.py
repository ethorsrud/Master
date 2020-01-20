# coding=utf-8
from torch import nn
import torch
import numpy as np

"""
Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
Progressive Growing of GANs for Improved Quality, Stability, and Variation.
Retrieved from http://arxiv.org/abs/1710.10196
"""

class ProgressiveDiscriminator(nn.Module):
	"""
	Discriminator module for implementing progressive GANS

	Attributes
	----------
	block : list
		List of `ProgressiveDiscriminatorBlock` which each represent one
		stage during progression
	cur_block : int
		Current stage of progression (from last to first)
	alpha : float
		Fading parameter. Defines how much of the input skips the current block

	Parameters
	----------
	blocks : int
		Number of progression stages
	"""
	def __init__(self,blocks,conditional=False):
		super(ProgressiveDiscriminator,self).__init__()
		self.blocks = nn.ModuleList(blocks)
		self.cur_block = len(self.blocks)-1
		self.alpha = 1.
		#To make sure labels arent added to the fourier discriminator
		self.conditional = conditional

	def forward(self,input):
		fade = False
		alpha = self.alpha
		#MAKE NUMPY ARRAY OF LABEL
		if self.conditional:
			orig_label = input[:,:,:,-1]
			orig_label_np = orig_label.cpu().detach().numpy()
			print(input.shape)

		for i in range(self.cur_block,len(self.blocks)):
			if alpha<1. and i==self.cur_block:
				tmp = self.blocks[i].fade_sequence(input)
				tmp = self.blocks[i+1].in_sequence(tmp)
				fade = True
			
			if fade and i==self.cur_block+1:
				input = alpha*input+(1.-alpha)*tmp
			
			if self.conditional and i!=self.cur_block:
				factor = orig_label.shape[-1]/input.shape[-1]
                #USE NUMPY ARRAY OF LABEL TO MAKE DOWNSAMPLED LABEL
				
				#label = np.zeros(shape=(input.shape[0],input.shape[1],input.shape[2],1))
				pass

			input = self.blocks[i](input,
								first=(i==self.cur_block))
			print(input.shape)
		return input

	def downsample_to_block(self,input,i_block):
		"""
		Scales down input to the size of current input stage.
		Utilizes `ProgressiveDiscriminatorBlock.fade_sequence` from each stage.

		Parameters
		----------
		input : autograd.Variable
			Input data
		i_block : int
			Stage to which input should be downsampled

		Returns
		-------
		output : autograd.Variable
			Downsampled data
		"""
		for i in range(i_block):
			input = self.blocks[i].fade_sequence(input)
		output = input
		return output

class ProgressiveGenerator(nn.Module):
	"""
	Generator module for implementing progressive GANS

	Attributes
	----------
	block : list
		List of `ProgressiveGeneratorBlock` which each represent one
		stage during progression
	cur_block : int
		Current stage of progression (from first to last)
	alpha : float
		Fading parameter. Defines how much of the second to last stage gets
		merged into the output.

	Parameters
	----------
	blocks : int
		Number of progression stages
	"""
	def __init__(self,blocks):
		super(ProgressiveGenerator,self).__init__()
		self.blocks = nn.ModuleList(blocks)
		self.cur_block = 0
		self.alpha = 1.

	def forward(self,input,label):
		#print("Label",label.shape)
		n_blocks = len(self.blocks)
		base = input.shape[1]
		fade = False
		alpha = self.alpha
		for i in range(0,self.cur_block+1):
			#Adding labels to input
			labels = np.zeros(shape=(input.shape[0],base*2**i))
			label_downsampled = np.floor(label/(2**(n_blocks-i))).astype(np.int)
			indexes = (np.arange(input.shape[0]).astype(np.int),label_downsampled)
			labels[indexes] = 1.
			if i==0:
				input = input[:,:,None]
				labels = labels[:,:,np.newaxis].astype(np.float32)
				labels = torch.from_numpy(labels).cuda()
				#print("i==0",input.shape,labels.shape)
				input = torch.cat((input,labels),dim=2)
			else:
				labels = labels[:,np.newaxis,:].astype(np.float32)
				labels = torch.from_numpy(labels).cuda()
				#print("i!=0",input.shape,labels.shape)
				input = torch.cat((input,labels),dim=1)

			

			input = self.blocks[i](input,last=(i==self.cur_block))

			if alpha<1. and i==self.cur_block-1:
				tmp = self.blocks[i].out_sequence(input)
				fade = True

		if fade:
			tmp = self.blocks[i-1].fade_sequence(tmp)
			input = alpha*input+(1.-alpha)*tmp
		return input

class ProgressiveDiscriminatorBlock(nn.Module):
	"""
	Block for one Discriminator stage during progression

	Attributes
	----------
	intermediate_sequence : nn.Sequence
		Sequence of modules that process stage
	in_sequence : nn.Sequence
		Sequence of modules that is applied if stage is the current input
	fade_sequence : nn.Sequence
		Sequence of modules that is used for fading input into stage
	"""
	def __init__(self,intermediate_sequence,in_sequence,fade_sequence):
		super(ProgressiveDiscriminatorBlock,self).__init__()
		self.intermediate_sequence = intermediate_sequence
		self.in_sequence = in_sequence
		self.fade_sequence = fade_sequence

	def forward(self,input,first=False):
		if first:
			input = self.in_sequence(input)
		out = self.intermediate_sequence(input)
		return out

class ProgressiveGeneratorBlock(nn.Module):
	"""
	Block for one Generator stage during progression

	Attributes
	----------
	intermediate_sequence : nn.Sequence
		Sequence of modules that process stage
	out_sequence : nn.Sequence
		Sequence of modules that is applied if stage is the current output
	fade_sequence : nn.Sequence
		Sequence of modules that is used for fading stage into output
	"""
	def __init__(self,intermediate_sequence,out_sequence,fade_sequence):
		super(ProgressiveGeneratorBlock,self).__init__()
		self.intermediate_sequence = intermediate_sequence
		self.out_sequence = out_sequence
		self.fade_sequence = fade_sequence

	def forward(self,input,last=False):
		out = self.intermediate_sequence(input)
		if last:
			out = self.out_sequence(out)
		return out
