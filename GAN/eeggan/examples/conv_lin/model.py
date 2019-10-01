#import braindecode
from torch import nn
from eeggan.modules.layers.reshape import Reshape,PixelShuffle2d
from eeggan.modules.layers.normalization import PixelNorm
from eeggan.modules.layers.weight_scaling import weight_scale
from eeggan.modules.layers.upsampling import CubicUpsampling1d,CubicUpsampling2d
from eeggan.modules.layers.stdmap import StdMap1d
from eeggan.modules.progressive import ProgressiveGenerator,ProgressiveGeneratorBlock,\
							ProgressiveDiscriminator,ProgressiveDiscriminatorBlock
from eeggan.modules.wgan import WGAN_I_Generator,WGAN_I_Discriminator
from torch.nn.init import calculate_gain

#INSTEAD OF kernel=5 and pad=2, originial: kernel=9 and pad=4
n_featuremaps = 25
#base = starting samples => base = input_size/(2**N_blocks)
base = int(1536/(2**6))
def create_disc_blocks(n_chans):
	def create_conv_sequence(in_filters,out_filters):
		return nn.Sequential(weight_scale(nn.Conv1d(in_filters,in_filters,5,padding=2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								weight_scale(nn.Conv1d(in_filters,out_filters,5,padding=2),
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
							  create_conv_sequence(n_featuremaps,n_featuremaps),
							  create_in_sequence(n_chans,n_featuremaps),
							  create_fade_sequence(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence(n_featuremaps,n_featuremaps),
							  create_in_sequence(n_chans,n_featuremaps),
							  create_fade_sequence(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							create_conv_sequence(n_featuremaps,n_featuremaps),
							create_in_sequence(n_chans,n_featuremaps),
							create_fade_sequence(2)
							)
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							create_conv_sequence(n_featuremaps,n_featuremaps),
							create_in_sequence(n_chans,n_featuremaps),
							create_fade_sequence(2)
							)
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence(n_featuremaps,n_featuremaps),
							  create_in_sequence(n_chans,n_featuremaps),
							  create_fade_sequence(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  nn.Sequential(StdMap1d(),
											create_conv_sequence(n_featuremaps+1,n_featuremaps),
											Reshape([[0],-1]),
											weight_scale(nn.Linear(n_featuremaps*base,1),
															gain=calculate_gain('linear'))),
							  create_in_sequence(n_chans,n_featuremaps),
							  None
							  )
	blocks.append(tmp_block)
	return blocks


def create_gen_blocks(n_chans,z_vars):
	def create_conv_sequence(in_filters,out_filters):
		return nn.Sequential(nn.Upsample(mode='linear',scale_factor=2),
								weight_scale(nn.Conv1d(in_filters,out_filters,5,padding=2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								PixelNorm(),
								weight_scale(nn.Conv1d(out_filters,out_filters,5,padding=2),
														gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								PixelNorm())
	def create_out_sequence(n_chans,in_filters):
		return nn.Sequential(weight_scale(nn.Conv1d(in_filters,n_chans,1),
														gain=calculate_gain('linear')),
								Reshape([[0],[1],[2],1]),
								PixelShuffle2d([1,n_chans]))
	def create_fade_sequence(factor):
		return nn.Upsample(mode='bilinear',scale_factor=(2,1))
	blocks = []
	#originally n_featuremaps*12
	tmp_block = ProgressiveGeneratorBlock(
								nn.Sequential(weight_scale(nn.Linear(z_vars,n_featuremaps*base),
														gain=calculate_gain('leaky_relu')),
												nn.LeakyReLU(0.2),
												Reshape([[0],n_featuremaps,-1]),
												create_conv_sequence(n_featuremaps,n_featuremaps)),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								create_fade_sequence(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence(n_featuremaps,n_featuremaps),
								create_out_sequence(n_chans,n_featuremaps),
								None
								)
	blocks.append(tmp_block)
	return blocks


class Generator(WGAN_I_Generator):
	def __init__(self,n_chans,z_vars):
		super(Generator,self).__init__()
		self.model = ProgressiveGenerator(create_gen_blocks(n_chans,z_vars))

	def forward(self,input):
		return self.model(input)

class Discriminator(WGAN_I_Discriminator):
	def __init__(self,n_chans):
		super(Discriminator,self).__init__()
		self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans))

	def forward(self,input):
		return self.model(input)
