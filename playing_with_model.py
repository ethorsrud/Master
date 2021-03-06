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
from matplotlib.collections import LineCollection

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

torch.cuda.set_device(0)

n_z = 128
datafreq = 30000
n_blocks = 3
t_multiple = 3
input_length = 2048#4096#8192
n_chans = 57
label_length = 20

generator = Generator(n_chans,128+input_length) #Channels, random vector input size
generator.train_init(alpha=1e-3,betas=(0.,0.99))
generator.load_model(os.path.join(model_path,"Progressive0.gen"),location="cuda:0")
i_block,fade_alpha = joblib.load(os.path.join(model_path,"Progressive0"+'.data'))

generator.model.cur_block = i_block
generator.model.alpha = fade_alpha

generator.cuda()

mean_std = np.load("real_mean_std_dataset_alpha.npy")
spike_mean = mean_std[0]
spike_std = mean_std[1]
rng = np.random.RandomState(0)

print("Mean:",spike_mean,"Std:",spike_std)

#n_spikes = #10 STATIC
for i in range(1):
    z_vars_im = rng.normal(0,1,size=(1,n_z)).astype(np.float32)
    labels = np.zeros(shape=(1,input_length))
    #Random number of spikes
    n_spikes = int(np.random.normal(spike_mean,spike_std))

    if n_spikes<0:
        n_spikes=0
    #Create n_spikes randomly times spikes
    random_times = np.random.randint(0,input_length-21,size=(n_spikes)).astype(np.int)
    for j in range(n_spikes):
        labels[0,random_times[j]:(random_times[j]+label_length)] = 1.

    labels = labels.astype(np.float32)
    z_vars_im = np.concatenate((z_vars_im,labels),axis=1)
    z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False).cuda()
    batch_fake = generator(z_vars)
    dataset = batch_fake.detach().cpu().numpy()

    dataset = dataset.squeeze()

    for j in range(57):
        plt.plot(dataset[:,j]-5*j,color="steelblue",linewidth=0.1)

        for t in random_times:
            plt.plot(np.arange(t,t+label_length),dataset[int(t):int(t)+label_length,j]-5*j,color="red",alpha=0.5,linewidth=0.1)

    plt.yticks([])
    plt.xticks(fontsize=15)
    plt.title("A view of all channels of a single time signal",fontsize=15)
    plt.xlabel("Time sample i",fontsize=15)
    plt.show()
