import os
import sys
import joblib
code_path = os.path.normpath(os.getcwd())
other_path = os.path.normpath(code_path+os.sep+os.pardir)
model_path = os.path.normpath(other_path+os.sep+"Models"+os.sep+"GAN")
sys.path.append(os.path.join(code_path,"GAN"))
sys.path.append(code_path)
sys.path.append("/home/eirith/.local/lib/python3.5/site-packages")
from eeggan.examples.conv_lin.model import Generator
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

torch.cuda.set_device(3)

n_z = 128
datafreq = 30000
n_blocks = 6
t_multiple = 3

generator = Generator(2,128) #Channels, random vector input size
generator.train_init(alpha=1e-3,betas=(0.,0.99))
generator.load_model(os.path.join(model_path,"Progressive0.gen"))
i_block,fade_alpha = joblib.load(os.path.join(model_path,"Progressive0"+'.data'))

generator.model.cur_block = i_block
generator.model.alpha = fade_alpha

generator.cuda()
print("Model loaded")

rng = np.random.RandomState(0)
z_vars_im = rng.normal(0,1,size=(1000,n_z)).astype(np.float32)
z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False).cuda()
batch_fake = generator(z_vars)

z_vars_im_longer = rng.normal(0,1,size=(1000,n_z*t_multiple)).astype(np.float32)
z_vars_longer = Variable(torch.from_numpy(z_vars_im_longer),requires_grad=False).cuda()
batch_fake_longer = generator(z_vars_longer)

plt.plot(batch_fake[0,0,:,0].detach().cpu().numpy())
plt.plot(batch_fake_longer[0,0,:,0].detach().cpu().numpy()+0.5)
plt.ylim(-1,1)
plt.legend(["Trained length","Extended length"])
plt.savefig("Loaded_GAN_time.png")
plt.close()

torch_fake_fft = np.swapaxes(torch.rfft(np.swapaxes(batch_fake.data.cpu(),2,3),1),2,3)
torch_fake_fft = torch.sqrt(torch_fake_fft[:,:,:,:,0]**2+torch_fake_fft[:,:,:,:,1]**2)
fake_amps = torch_fake_fft.data.cpu().numpy().mean(axis=0).squeeze()
freqs_tmp = np.fft.rfftfreq(batch_fake.shape[2],d=1/(datafreq/np.power(2,n_blocks-1-i_block)))

print(batch_fake.shape)
print(batch_fake_longer.shape)
batch_fake_longer.view(1000*t_multiple,1,batch_fake.shape[2],2)
print(batch_fake_longer.shape)
torch_fake_longer_fft = np.swapaxes(torch.rfft(np.swapaxes(batch_fake_longer.data.cpu(),2,3),1),2,3)
torch_fake_longer_fft = torch.sqrt(torch_fake_longer_fft[:,:,:,:,0]**2+torch_fake_longer_fft[:,:,:,:,1]**2)
fake_amps_longer = torch_fake_longer_fft.data.cpu().numpy().mean(axis=0).squeeze()
freqs_tmp_longer = np.fft.rfftfreq(batch_fake_longer.shape[2],d=1/(datafreq/np.power(2,n_blocks-1-i_block)))

plt.plot(freqs_tmp,fake_amps[:,0])
plt.plot(freqs_tmp_longer,fake_amps_longer[:,0]/3)
plt.legend(["Trained length","Extended length"])
plt.savefig("Loaded_GAN_fft.png")
plt.close()