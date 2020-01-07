# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import eeggan.util as utils
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

class GAN_Module(nn.Module):
	"""
	Parent module for different GANs

	Attributes
	----------
	optimizer : torch.optim.Optimizer
		Optimizer for training the model parameters
	loss : torch.nn.Loss
		Loss function
	"""
	def __init__(self):
		super(GAN_Module, self).__init__()

		self.did_init_train = False

	def save_model(self,fname):
		"""
		Saves `state_dict` of model and optimizer

		Parameters
		----------
		fname : str
			Filename to save
		"""
		cuda = False
		if next(self.parameters()).is_cuda: cuda = True
		cpu_model = self.cpu()
		model_state = cpu_model.state_dict()
		opt_state = cpu_model.optimizer.state_dict()

		torch.save((model_state,opt_state,self.did_init_train),fname)
		if cuda:
			self.cuda()

	def load_model(self,fname,location=None):
		"""
		Loads `state_dict` of model and optimizer

		Parameters
		----------
		fname : str
			Filename to load from
		"""
		model_state,opt_state,self.did_init_train = torch.load(fname,map_location=location)

		self.load_state_dict(model_state)
		self.optimizer.load_state_dict(opt_state)


class GAN_Discriminator(GAN_Module):
	"""
	Vanilla GAN discriminator

	References
	----------
	Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D.,
	Ozair, S., … Bengio, Y. (2014). Generative Adversarial Networks.
	Retrieved from http://arxiv.org/abs/1406.2661
	"""
	def __init__(self):
		super(GAN_Discriminator, self).__init__()

	def train_init(self,alpha=1e-4,betas=(0.5,0.9)):
		"""
		Initialize Adam optimizer and BCE loss for discriminator

		Parameters
		----------
		alpha : float, optional
			Learning rate for Adam
		betas : (float,float), optional
			Betas for Adam
		"""
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = torch.nn.BCELoss()
		self.did_init_train = True

	def pre_train(self):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

	def update_parameters(self):
		self.optimizer.step()

	def train_batch(self, batch_real, batch_fake):
		"""
		Train discriminator for one batch of real and fake data

		Parameters
		----------
		batch_real : autograd.Variable
			Batch of real data
		batch_fake : autograd.Variable
			Batch of fake data

		Returns
		-------
		loss_real : float
			BCE loss for real data
		loss_fake : float
			BCE loss for fake data
		"""
		self.pre_train()

		ones_label = torch.ones(batch_real.size(0),1)
		zeros_label = torch.zeros(batch_fake.size(0),1)

		ones_label = Variable(ones_label)
		zeros_label = Variable(zeros_label)

		batch_real,ones_label,zeros_label = utils.cuda_check([batch_real,
															  ones_label,
															  zeros_label])

		# Compute output and loss
		fx_real = self.forward(batch_real)
		loss_real = self.loss.forward(fx_real,ones_label)
		loss_real.backward()
		fx_fake = self.forward(batch_fake)
		loss_fake = self.loss.forward(fx_fake,zeros_label)
		loss_fake.backward()

		self.update_parameters()

		loss_real = loss_real.data[0]
		loss_fake = loss_fake.data[0]
		return loss_real,loss_fake # return loss


class GAN_Generator(GAN_Module):
	"""
	Vanilla GAN generator

	References
	----------
	Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D.,
	Ozair, S., … Bengio, Y. (2014). Generative Adversarial Networks.
	Retrieved from http://arxiv.org/abs/1406.2661
	"""
	def __init__(self):
		super(GAN_Generator, self).__init__()

	def train_init(self,alpha=1e-4,betas=(0.5,0.9)):
		"""
		Initialize Adam optimizer and BCE loss for generator

		Parameters
		----------
		alpha : float, optional
			Learning rate for Adam
		betas : (float,float), optional
			Betas for Adam
		"""
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = torch.nn.BCELoss()
		self.did_init_train = True

	def pre_train(self,discriminator):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

	def update_parameters(self):
		"""
		for yolo in self.parameters():
			print(yolo.shape)
			yolo_investigation = yolo.detach().cpu().numpy()
			try:
				yolo_grad_investigation = yolo.grad.data.cpu().numpy()
			except:
				yolo_grad_investigation = np.array([0])
			if len(yolo_grad_investigation)!=1:
				print("Min param =",np.min(yolo_investigation),"Max param =",np.max(yolo_investigation))
			print("Parameter finite?",np.all(np.isfinite(yolo_investigation)))
		"""
		self.optimizer.step()
		"""
		for yolo in self.parameters():
			print(yolo.shape)
			yolo_investigation = yolo.detach().cpu().numpy()
			print("Parameter still finite?",np.all(np.isfinite(yolo_investigation)))
		"""
	def train_batch(self, batch_noise, discriminator):
		"""
		Train generator for one batch of latent noise

		Parameters
		----------
		batch_noise : autograd.Variable
			Batch of latent noise
		discriminator : nn.Module
			Discriminator to evaluate realness of generated data

		Returns
		-------
		loss : float
			BCE loss against evaluation of discriminator of generated samples
			to be real
		"""
		self.pre_train(discriminator)

		# Generate and discriminate
		gen = self.forward(batch_noise)
		disc = discriminator(gen)

		ones_label = torch.ones(disc.size())

		ones_label = Variable(ones_label)

		batch_noise,ones_label = utils.cuda_check([batch_noise,ones_label])

		loss = self.loss.forward(disc,ones_label)

		# Backprop gradient
		loss.backward()

		# Update parameters
		self.update_parameters()

		loss = loss.data[0]
		return loss # return loss


class GAN_Discriminator_SoftPlus(GAN_Module):
	"""
	Improved GAN discriminator

	References
	----------
	Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
	Chen, X. (2016). Improved Techniques for Training GANs. Learning;
	Computer Vision and Pattern Recognition; Neural and Evolutionary Computing.
	Retrieved from http://arxiv.org/abs/1606.03498
	"""
	def __init__(self):
		super(GAN_Discriminator_SoftPlus, self).__init__()

	def train_init(self,alpha=1e-4,betas=(0.5,0.9)):
		"""
		Initialize Adam optimizer for discriminator

		Parameters
		----------
		alpha : float, optional
			Learning rate for Adam
		betas : (float,float), optional
			Betas for Adam
		"""
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = None
		self.did_init_train = True

	def pre_train(self):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in self.parameters():
			p.requires_grad = True

	def update_parameters(self):
		self.optimizer.step()

	def train_batch(self, batch_real, batch_fake):
		"""
		Train discriminator for one batch of real and fake data

		Parameters
		----------
		batch_real : autograd.Variable
			Batch of real data
		batch_fake : autograd.Variable
			Batch of fake data

		Returns
		-------
		loss_real : float
			BCE loss for real data
		loss_fake : float
			BCE loss for fake data
		"""
		self.pre_train()

		# Compute output and loss
		fx_real = self.forward(batch_real)
		loss_real = F.softplus(-fx_real).mean()
		loss_real.backward()

		fx_fake = self.forward(batch_fake)
		loss_fake = F.softplus(fx_fake).mean()
		loss_fake.backward()

		self.update_parameters()

		loss_real = loss_real.data[0]
		loss_fake = loss_fake.data[0]
		return loss_real,loss_fake # return loss


class GAN_Generator_SoftPlus(GAN_Module):
	"""
	Improved GAN generator

	References
	----------
	Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &
	Chen, X. (2016). Improved Techniques for Training GANs. Learning;
	Computer Vision and Pattern Recognition; Neural and Evolutionary Computing.
	Retrieved from http://arxiv.org/abs/1606.03498
	"""
	def __init__(self):
		super(GAN_Generator_SoftPlus, self).__init__()

	def train_init(self,alpha=1e-4,betas=(0.5,0.9)):
		"""
		Initialize Adam optimizer for generator

		Parameters
		----------
		alpha : float, optional
			Learning rate for Adam
		betas : (float,float), optional
			Betas for Adam
		"""
		self.optimizer = optim.Adam(self.parameters(),lr=alpha,betas=betas)
		self.loss = None
		self.did_init_train = True

	def pre_train(self,discriminator):
		if not self.did_init_train:
			self.train_init()

		self.zero_grad()
		self.optimizer.zero_grad()
		for p in discriminator.parameters():
			p.requires_grad = False  # to avoid computation

	def update_parameters(self):
		self.optimizer.step()

	def train_batch(self, batch_noise, discriminator):
		"""
		Train generator for one batch of latent noise

		Parameters
		----------
		batch_noise : autograd.Variable
			Batch of latent noise
		discriminator : nn.Module
			Discriminator to evaluate realness of generated data

		Returns
		-------
		loss : float
			Loss of evaluation of discriminator of generated samples
			to be real
		"""
		self.pre_train(discriminator)

		# Generate and discriminate
		gen = self.forward(batch_noise)
		disc = discriminator(gen)

		loss = F.softplus(-disc).mean()

		# Backprop gradient
		loss.backward()

		# Update parameters
		self.update_parameters()

		loss = loss.data[0]
		return loss # return loss
