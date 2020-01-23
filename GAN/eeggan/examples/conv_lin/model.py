#import braindecode
from torch import nn
from eeggan.modules.layers.reshape import Reshape,PixelShuffle2d
from eeggan.modules.layers.normalization import PixelNorm
from eeggan.modules.layers.weight_scaling import weight_scale
from eeggan.modules.layers.upsampling import CubicUpsampling1d,CubicUpsampling2d,upsample_layer
from eeggan.modules.layers.stdmap import StdMap1d
from eeggan.modules.layers.fouriermap import FFTMap1d
from eeggan.modules.progressive import ProgressiveGenerator,ProgressiveGeneratorBlock,\
							ProgressiveDiscriminator,ProgressiveDiscriminatorBlock
from eeggan.modules.wgan import WGAN_I_Generator,WGAN_I_Discriminator
from torch.nn.init import calculate_gain
from eeggan.modules.layers.xray import xrayscanner

#INSTEAD OF kernel=5 and pad=2, originial: kernel=9 and pad=4
n_featuremaps = 10#25
#base = starting samples => base = input_size/(2**N_blocks)
base = int(8192/(2**6))#int(1536/(2**6))
"""
Align corners-error
UserWarning: Default upsampling behavior when mode=linear is changed to align_corners=False since 0.4.0.
Please specify align_corners=True if the old behavior is desired.
See the documentation of nn.Upsample for details.
"""
Align = False
"""
REMOVED (after first leakyrelu)
								weight_scale(nn.Conv1d(in_filters,out_filters,5,padding=2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),

Also changed in_filters,infilters to ----> in_filters,out_filters

"""


def create_disc_blocks(n_chans,base,conditional):
	def create_conv_sequence(in_filters,out_filters):
		return nn.Sequential(weight_scale(nn.Conv1d(in_filters,out_filters,9,padding=4),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),

								weight_scale(nn.Conv1d(out_filters,out_filters,2,stride=2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2))
	def create_in_sequence(n_chans,out_filters):
		return nn.Sequential(weight_scale(nn.Conv2d(1,out_filters,(1,n_chans)),
														gain=calculate_gain('leaky_relu')),
								Reshape([[0],[1],[2]]),
								nn.LeakyReLU(0.2))
	def create_fade_sequence(factor):
		return nn.AvgPool2d((factor,1),stride=(factor,1))
		
	blocks = []
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
							  create_in_sequence(n_chans,n_featuremaps+conditional),
							  create_fade_sequence(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
							  create_in_sequence(n_chans,n_featuremaps+conditional),
							  create_fade_sequence(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
							create_in_sequence(n_chans,n_featuremaps+conditional),
							create_fade_sequence(2)
							)
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
							create_in_sequence(n_chans,n_featuremaps+conditional),
							create_fade_sequence(2)
							)
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
							  create_in_sequence(n_chans,n_featuremaps+conditional),
							  create_fade_sequence(2)
							  )
	blocks.append(tmp_block)


	#Removed StdMap1d() before create_conv_sequence and reduced (n_featuremaps+1,n_featuremaps) to (n_featuremaps,n_featuremaps)

	tmp_block = ProgressiveDiscriminatorBlock(
							  nn.Sequential(StdMap1d(),
											create_conv_sequence(n_featuremaps+1+conditional,n_featuremaps),
											Reshape([[0],-1]),
											weight_scale(nn.Linear((n_featuremaps)*base,1),
															gain=calculate_gain('linear'))),
							  create_in_sequence(n_chans,n_featuremaps+conditional),
							  None
							  )
	blocks.append(tmp_block)
	return blocks

"""
REMOVED (after Pixelnorm)
								weight_scale(nn.Conv1d(out_filters,out_filters,5,padding=2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								PixelNorm()
"""
def create_gen_blocks(n_chans,z_vars,conditional):
	def create_conv_sequence(in_filters,out_filters):
		return nn.Sequential(upsample_layer(mode='linear',scale_factor=2),
								weight_scale(nn.Conv1d(in_filters,out_filters,9,padding=4),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								PixelNorm(),
								)

	def create_out_sequence(n_chans,in_filters):
		return nn.Sequential(weight_scale(nn.Conv1d(in_filters,n_chans,1),
														gain=calculate_gain('linear')),
								Reshape([[0],[1],[2],1]),
								PixelShuffle2d([1,n_chans]))
	def create_fade_sequence(factor):
		#return nn.Upsample(mode='bilinear',scale_factor=(2,1))
		return upsample_layer(mode='bilinear',scale_factor=(2,1))
	blocks = []
	#originally n_featuremaps*12
	
	#Original No reshape, only one linear layer z_vars,base*featuremaps
	tmp_block = ProgressiveGeneratorBlock(
								nn.Sequential(weight_scale(nn.Linear(z_vars,base*(n_featuremaps+conditional)),
														gain=calculate_gain('leaky_relu')),
												nn.LeakyReLU(0.2),
												
												Reshape([[0],n_featuremaps+conditional,-1]),
												create_conv_sequence(n_featuremaps+conditional,n_featuremaps)),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	

	"""
	tmp_block = ProgressiveGeneratorBlock(
								nn.Sequential(Reshape([[0],2,-1]),
								weight_scale(nn.Conv1d(2,n_featuremaps,1001,padding=500),
														gain=calculate_gain('leaky_relu')),
												nn.LeakyReLU(0.2),
								weight_scale(nn.Conv1d(n_featuremaps,n_featuremaps,9,padding=4),
														gain=calculate_gain('leaky_relu')),
												nn.LeakyReLU(0.2),
												Reshape([[0],n_featuremaps,-1]),
												create_conv_sequence(n_featuremaps,n_featuremaps),
								),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	"""
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps+conditional,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								None
								)
	blocks.append(tmp_block)
	return blocks


class Generator(WGAN_I_Generator):
	def __init__(self,n_chans,z_vars):
		super(Generator,self).__init__()
		self.model = ProgressiveGenerator(create_gen_blocks(n_chans,z_vars,conditional=False),conditional=False)

	def forward(self,input):
		return self.model(input)

class Discriminator(WGAN_I_Discriminator):
	def __init__(self,n_chans):
		super(Discriminator,self).__init__()
		self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans,base,conditional=False),conditional=False)

	def forward(self,input):
		return self.model(input)

class Fourier_Discriminator(WGAN_I_Discriminator):
	def __init__(self,n_chans):
		super(Fourier_Discriminator,self).__init__()
		self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans,int(base/2),conditional=False),conditional=False)

	def forward(self,input):
		return self.model(input)

class AC_Discriminator(WGAN_I_Discriminator):
	def __init__(self,n_chans):
		super(AC_Discriminator,self).__init__()
		self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans,int(base/2),conditional=False),conditional=False)

	def forward(self,input):
		return self.model(input)
