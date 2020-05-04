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

torch.cuda.set_device(2)

kilosort_path = os.path.normpath(os.getcwd()+4*(os.sep+os.pardir)+os.sep+"shared"+os.sep+"users"+os.sep+"eirith"+os.sep+"kilosort2_results"+os.sep)
#templates = np.load(os.path.normpath(kilosort_path+os.sep+"templates.npy")).astype(np.float32) #[nTemplates,nTimePoints,nTempChannels]
templates = np.load("templates_ch120_ch180.npy").astype(np.float32)
templates = np.mean(templates,axis=2)

n_z = 128
datafreq = 30000
n_blocks = 6
t_multiple = 3
input_length = 2048#4096#8192
n_chans = 57
label_length = 20

generator = Generator(n_chans,128+input_length) #Channels, random vector input size
generator.train_init(alpha=1e-3,betas=(0.,0.99))
generator.load_model(os.path.join(model_path,"Progressive0.gen"),location="cuda:2")
i_block,fade_alpha = joblib.load(os.path.join(model_path,"Progressive0"+'.data'))

generator.model.cur_block = i_block
generator.model.alpha = fade_alpha

generator.cuda()

mean_std = np.load("real_mean_std_dataset_alpha.npy")
spike_mean = mean_std[0]
spike_std = mean_std[1]
rng = np.random.RandomState(0)

print("Mean:",spike_mean,"Std:",spike_std)

z_vars_im = rng.normal(0,1,size=(768,n_z)).astype(np.float32)
labels = np.zeros(shape=(768,input_length))
#labels_ones = np.zeros(shape=(768,input_length))
for i in range(768):
    #Random number of spikes
    n_spikes = int(np.random.normal(spike_mean,spike_std))
    if n_spikes<0:
        n_spikes=0
    #Create n_spikes randomly times spikes
    random_times = np.random.randint(0,input_length-21,size=(n_spikes)).astype(np.int)
    #random_templates = np.random.randint(0,templates.shape[0],size=(n_spikes)).astype(np.int)
    for j in range(n_spikes):
        labels[i,random_times[j]:(random_times[j]+label_length)] = 1.
        #labels[i,(random_times[j]-41):(random_times[j]+41)] = templates[random_templates[j],:]

labels = labels.astype(np.float32)
#labels_ones = labels_ones.astype(np.float32)
z_vars_im = np.concatenate((z_vars_im,labels),axis=1)
z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False).cuda()
batch_fake = generator(z_vars)
dataset = batch_fake.detach().cpu().numpy()

dataset = dataset.squeeze()

dataset = dataset.reshape((input_length*768,n_chans))
labels = labels.reshape(-1)
#labels_ones = labels_ones.reshape(-1)
#spike_times = np.where(labels_ones==1.)[0]

np.save("fake_dataset_nofourier_ch120_ch180_alpha.npy",dataset)
#np.save("fake_dataset_ch120_ch160_labels_ones_57.npy",spike_times)
np.save("fake_dataset_nofourier_ch120_ch180_labels_alpha.npy",labels)
"""
rng = np.random.RandomState(0)
z_vars_im = rng.normal(0,1,size=(500,n_z)).astype(np.float32)
random_times = np.linspace(0,input_length-80,500).astype(np.int)
random_times = (np.zeros(500)+input_length/2).astype(np.int)
#labels = np.zeros(shape=(500,n_z))
labels = np.zeros(shape=(500,input_length))
for i in range(500):
    labels[i,random_times[i]:(random_times[i]+1)] = 1.
#label_downsampled = np.floor(random_times/(2**n_blocks)).astype(np.int)
#indexes = (np.arange(500).astype(np.int),label_downsampled)
#labels[indexes] = 1.
labels = labels.astype(np.float32)
z_vars_im = np.concatenate((z_vars_im,labels),axis=1)

z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False).cuda()

batch_fake = generator(z_vars)

mid = int(batch_fake.shape[2]//2)
for i in range(100):
    plt.plot(np.arange(mid-40,mid+40),batch_fake[i,0,(mid-40):(mid+40),0].detach().cpu().numpy(),linewidth=0.3,alpha=0.5)
plt.savefig("Checking_mode_collapse_ch0.png",dpi=500)
plt.close()
"""
"""
for ch in range(11):
    template_extended = np.zeros(input_length)
    peak_length = templates.shape[1]
    template_extended[(int(input_length//2)-45):(int(input_length//2+peak_length)-45)] = templates[0,:,ch]*0.4
    for i in range(200):
        plt.plot(np.arange(4025,4175),batch_fake[i,0,4025:4175,ch].detach().cpu().numpy()-np.mean(batch_fake[i,0,4025:4175,ch].detach().cpu().numpy()),linewidth=0.3,alpha=0.5)
    plt.plot(np.arange(4025,4175),template_extended[4025:4175],label="Template",linewidth = 1.5,alpha=0.45)
    plt.legend()
    plt.title("200 signals where the label is set \n to create spike of template 0 in the middle \n CHANNEL %i"%ch)
    plt.xlabel("Sample i of 8192 total")
    plt.savefig("Block_5_spike_ch%i.png"%ch,dpi=300)
    plt.close()
"""

"""
for block in range(i_block+1):
    generator.model.cur_block = block
    batch_fake = generator(z_vars)
    for i in range(128):
        plt.plot(batch_fake[i,0,:,0].detach().cpu().numpy()+0.5*i,linewidth=0.5)
    plt.savefig("Label_swipe_block_%i.png"%block,dpi=800)
    plt.close()
"""
#z_vars_im_longer = rng.normal(0,1,size=(400,n_z*t_multiple)).astype(np.float32)
#z_vars_longer = Variable(torch.from_numpy(z_vars_im_longer),requires_grad=False).cuda()
#batch_fake_longer = generator(z_vars_longer)



"""
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
batch_fake_longer.view(400*t_multiple,1,batch_fake.shape[2],2)
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
"""