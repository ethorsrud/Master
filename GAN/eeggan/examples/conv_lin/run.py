#%load_ext autoreload
#%autoreload 2
import os
import joblib
import sys
#import mne
code_path = os.path.normpath(os.getcwd()+4*(os.sep+os.pardir))
other_path = os.path.normpath(code_path+os.sep+os.pardir)
sys.path.append(os.path.join(code_path,"GAN"))
sys.path.append(code_path)
sys.path.append("/home/eirith/.local/lib/python3.5/site-packages")
#sys.path.append("/usr/local/lib/python3.5/dist-packages")
from braindecode.datautil.iterators import get_balanced_batches
from eeggan.examples.conv_lin.model import Generator,Discriminator,Fourier_Discriminator,AC_Discriminator
from eeggan.util import weight_filler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
from scipy.signal import butter,lfilter
from  datetime import datetime
#from torchviz import make_dot
from my_utils import functions
from scipy import signal
from scipy.fftpack import fft
from scipy import fftpack
import seaborn as sns
import json
from skimage.measure import block_reduce

#plt.switch_backend('agg')
#Error tracebacking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
cuda_device = 3
torch.cuda.set_device(cuda_device)

n_critic = 1
n_gen = 1
n_batch = 64#56#64
input_length = 4096#8192#10240#12288#30720#1536#768
jobid = 0
n_samples = 768 #Samples from dataset
conditional = True

n_z = 128#200
lr = 0.001#0.001
n_blocks = 6
rampup = 1000#400.#2000.
block_epochs = [1000,2000,2000,2000,2000,2000]#[2000,4000,4000,4000,4000,4000]

task_ind = 0#subj_ind

config_data = {"n_critic":n_critic,
                "n_gen":n_gen,
                "n_batch":n_batch,
                "input_length":input_length,
                "n_samples":n_samples,
                "conditional":conditional,
                "n_z":n_z,
                "lr":lr,
                "rampup":rampup,
                "block_epochs":block_epochs,
                "cuda_device:":cuda_device}

modelpath = os.path.normpath(other_path+os.sep+"Models"+os.sep+"GAN")
outputpath = os.path.normpath(other_path+os.sep+"Output"+os.sep+"GAN")

with open(os.path.normpath(modelpath+os.sep+"config.txt"),"w") as fp:
    json.dump(config_data,fp,indent=4)

#Spike data start
kilosort_path = os.path.normpath(os.getcwd()+7*(os.sep+os.pardir)+os.sep+"shared"+os.sep+"users"+os.sep+"eirith"+os.sep+"kilosort2_results"+os.sep)
dat_path = os.path.normpath(kilosort_path+os.sep+os.pardir+os.sep+"continuous.dat")

n_channels_dat = 384
data_len = 112933688832
dtype = 'int16'
offset = 0
sample_rate = 30000
hp_filtered = False

spike_data = np.memmap(dat_path, dtype, "r", offset, (data_len//n_channels_dat,n_channels_dat))
spike_data_small = spike_data[:input_length*n_samples,120:180]
train = spike_data_small.reshape((n_samples,input_length,60))[:,np.newaxis,:,:]
channel_map = np.load(code_path+os.sep+"channel_map_ch120_ch180.npy").astype(np.uint32)
#remove bad channels
train = train[:,:,:,channel_map[:,0]]

#np.save("spike_data_ch0_ch60.npy",spike_data_small)

#quit()
#FILTERING
#b,a = butter(10,6000/(0.5*sample_rate),btype="low")
#train = lfilter(b,a,train,axis=2)

"""
train_new = []
for i in range(int(spike_data_small.shape[0]/input_length)):
    train_new.append(spike_data_small[i*input_length:i*input_length+input_length])
print("creating array")
train_new = np.array(train_new)
train = train_new[:,:,:,np.newaxis]
train = np.swapaxes(train,1,2)
train = np.swapaxes(train,1,3)
#Only first channel
#train = train[:,:,:,0][:,:,:,np.newaxis]
"""
n_chans = train.shape[3]#+1 # +1 FOR CONDITIONAL
print("Number of channels:",n_chans)
print(train.shape)
#Spike data end
train = train.astype(np.float32)


np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)

datafreq = 30000#500#250#128 #hz
"""
data = os.path.normpath(other_path+os.sep+"Dataset"+os.sep+"All_channels_500Hz.npy")
#data = os.path.normpath(other_path+os.sep+"Dataset"+os.sep+"Two_channels_500hz.npy")
train = np.load(data).astype(np.float32)
train_new = []
for i in range(int(train.shape[0]/input_length)):
    train_new.append(train[i*input_length:i*input_length+input_length])
train_new = np.array(train_new)
train = train_new[:,:,:,np.newaxis]
train = np.swapaxes(train,1,2)
train = np.swapaxes(train,1,3)
#Only first channel
#train = train[:,:,:,0][:,:,:,np.newaxis]
n_chans = train.shape[3]
print("Number of channels:",n_chans)
print(train.shape)
"""
label_length = 1#1#80#1
"""
peak = np.linspace(0,2*np.pi,80)
peak = np.sin(peak)*200
#peak+=np.random.normal(size=(80))*70

#peak_train = train.copy()
time_labels = np.zeros(shape=(n_samples,1,input_length,1))
#train = np.concatenate((train,peak_train),axis=0)
#Placing random peaks
for i in range(n_samples):
    peak_location = np.random.randint(0,input_length-80)
    time_labels[i,0,peak_location:(peak_location+label_length),0] = 1
    train[i,0,(peak_location):(peak_location+80),0] += peak
"""
train = train-np.mean(train,axis=(0,2)).squeeze()#-train.mean()
train = train/np.std(train,axis=(0,2)).squeeze()#train.std()
train = train/np.max(np.abs(train)).squeeze()#np.max(np.abs(train),axis=(0,2)).squeeze()#np.abs(train).max()

#spike_times = np.load(os.path.normpath(kilosort_path+os.sep+"spike_times.npy")).astype(np.uint64) #[nSpikes,]
#spike_templates = np.load(os.path.normpath(kilosort_path+os.sep+"spike_templates.npy")).astype(np.uint32) #[nSpikes,]
#selected_template = 0
#temp_index = np.where(spike_templates==selected_template)[0]
spike_times = np.load(code_path+os.sep+"spike_times_ch120_ch180.npy").astype(np.uint64)
#spike_templates = np.load(code_path+os.sep+"spike_templates_ch120_ch180.npy").astype(np.uint32)
#templates = np.load(code_path+os.sep+"templates_ch120_ch180.npy").astype(np.float32)
#templates = (templates-np.mean(templates))/(np.std(templates))

time_labels = np.zeros(shape=(n_samples,1,input_length,1)).astype(np.float32)
#template_labels = np.zeros(shape=(n_samples,1,600,1))
#conv_labels = np.zeros(shape=(n_samples,1,input_length,n_chans)).astype(np.float32)
#conv_labels = np.zeros(shape=(n_samples,1,input_length,1)).astype(np.float32)
#conv_labels = rng.normal(0,1,size=(n_samples,1,input_length,n_chans)).astype(np.float32)
#Only spikes with selected template
#spike_times = spike_times[temp_index]
#mask
spike_times = spike_times[spike_times<(input_length*n_samples)]

mask = spike_times<(input_length*n_samples)
#mask = np.where(mask==1)
#spike_templates = spike_templates[mask]
#spike_templates = spike_templates[:,0]

#templates_new = np.mean(templates,axis=2)

for i in range(spike_times.shape[0]):
    cur_sample = int(spike_times[i]//input_length)
    cur_ind = int(spike_times[i]%input_length)
    time_labels[cur_sample,0,cur_ind:(cur_ind+label_length),0] = 1.
    template_length = input_length-cur_ind#82-((cur_ind+41)-input_length)
    if cur_ind>41 and cur_ind<(input_length-41):
        time_labels[cur_sample,0,cur_ind:(cur_ind+label_length),0] = 1.
        #conv_labels[cur_sample,0,(cur_ind-41):(cur_ind+41),:] = templates[spike_templates[i],:,:].astype(np.float32)
        #conv_labels[cur_sample,0,(cur_ind-41):(cur_ind+41),0] = templates_new[spike_templates[i],:].astype(np.float32)

#conv_labels = np.mean(conv_labels,axis=3)[:,:,:,np.newaxis]


n_spikes_per_samp = np.sum(time_labels,axis=2).squeeze()/label_length
spikes_mean = np.mean(n_spikes_per_samp)
spikes_std = np.sqrt(np.mean((n_spikes_per_samp-spikes_mean)**2))
print("Spikes_mean",spikes_mean,"Spikes_std",spikes_std)
np.save("real_mean_std_dataset.npy",np.array([spikes_mean,spikes_std]))

"""
n_templates_per_channel = np.sum(template_labels,axis=2).squeeze()
template_mean = np.mean(n_templates_per_channel)
template_std = np.sqrt(np.mean((n_templates_per_channel-template_mean)**2))
print("Templates_mean",template_mean,"Templates_std",template_std)
np.save("real_mean_std_templates.npy",np.array([template_mean,template_std]))
"""

#train = np.concatenate((train,time_labels),axis=3).astype(np.float32)
train = np.concatenate((train,time_labels),axis=3).astype(np.float32)
print("train_shape",train.shape)
fft_train = np.real(np.fft.rfft(train,axis=2))**2#np.abs(np.fft.rfft(train,axis=2))
#fft_train = np.log(fft_train)
#fft_mean = fft_train.mean()
#fft_std = fft_train.std()
#fft_max = np.abs(fft_train).max()



modelname = 'Progressive%s'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

#generator = Generator(n_chans,n_z*(1+conditional))
generator = Generator(n_chans,n_z+input_length)
discriminator = Discriminator(n_chans+1)
fourier_discriminator = Fourier_Discriminator(n_chans+1)
AC_discriminator = AC_Discriminator(n_chans)

generator.train_init(alpha=lr,betas=(0.,0.99))
discriminator.train_init(alpha=lr,betas=(0.,0.99),eps_center=0.001,
                        one_sided_penalty=True,distance_weighting=True)
fourier_discriminator.train_init(alpha=lr,betas=(0.,0.99),eps_center=0.001,
                        one_sided_penalty=True,distance_weighting=True)
AC_discriminator.train_init(alpha=lr,betas=(0.,0.99),eps_center=0.001,
                        one_sided_penalty=True,distance_weighting=True)
generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)
fourier_discriminator = fourier_discriminator.apply(weight_filler)
AC_discriminator = AC_discriminator.apply(weight_filler)

i_block_tmp = 0
i_epoch_tmp = 0
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = n_blocks-1-i_block_tmp
fourier_discriminator.model.cur_block = n_blocks-1-i_block_tmp
AC_discriminator.model.cur_block = n_blocks-1-i_block_tmp
fade_alpha = 1.
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha
fourier_discriminator.model.alpha = fade_alpha
AC_discriminator.model.alpha = fade_alpha

generator = generator.cuda()
discriminator = discriminator.cuda()
fourier_discriminator = fourier_discriminator.cuda()
AC_discriminator = AC_discriminator.cuda()
"""
#LOAD
try:
    generator.load_model(os.path.join(modelpath,modelname%jobid+'.gen'))
    discriminator.load_model(os.path.join(modelpath,modelname%jobid+'.disc'))
    generator.eval()
    discriminator.eval()
    i_block_tmp,i_epoch_tmp,losses_d,losses_g = joblib.load(os.path.join(modelpath,modelname%jobid+'_.data'))
    print("Model lock'n'loaded")

except:
    print("No model found, creating new")
    pass
"""
generator.train()
discriminator.train()
fourier_discriminator.train()
AC_discriminator.train()

losses_d = []
losses_g = []

losses_fourier = []

i_epoch = 0
z_vars_im = rng.normal(0,1,size=(700,n_z)).astype(np.float32)

#Conditional
if conditional:
    ##random_times_im = np.random.randint(0,input_length-80,size=(700)).astype(np.int)
    labels_im = np.zeros(shape=(700,input_length))
    #labels_im_new = np.zeros(shape=(700,input_length,n_chans))
    #labels_im_new = np.zeros(shape=(700,input_length))
    #labels_im_new = rng.normal(0,1,size=(700,input_length,n_chans))
    for i in range(700):
        #Random number of spikes
        n_spikes = int(np.random.normal(spikes_mean,spikes_std))
        if n_spikes<0:
            n_spikes=0
        #Create n_spikes randomly timed spikes
        random_times_im = np.random.randint(41,input_length-41,size=(n_spikes)).astype(np.int)
        #random_templates_im = np.random.randint(0,templates.shape[0],size=(n_spikes)).astype(np.int)
        for j in range(n_spikes):
            labels_im[i,random_times_im[j]:(random_times_im[j]+label_length)] = 1.
            #labels_im_new[i,(random_times_im[j]-41):(random_times_im[j]+41),:] = templates[random_templates_im[j],:,:]
            #labels_im_new[i,(random_times_im[j]-41):(random_times_im[j]+41)] = templates_new[random_templates_im[j],:]
    #index_im = np.where(labels_im==1.)
    #index_im = (index_im[0],np.floor(index_im[1]/(2**6)).astype(np.int))
    #labels_im = np.zeros(shape=(1000,n_z))
    #labels_im[index_im] = 1.
    labels_im = labels_im.astype(np.float32)
    #labels_im_new = labels_im_new.astype(np.float32)
    #labels_im = np.mean(labels_im_new,axis=2)
    z_vars_im = np.concatenate((z_vars_im,labels_im),axis=1)

#random_times_im = np.random.randint(0,n_z,size=(1000))
#z_vars_im_label[np.arange(1000),random_times_im] = 1.
#z_vars_im_label = z_vars_im_label.astype(np.float32)
#z_vars_im = z_vars_im[:,:,np.newaxis]
#z_vars_im_label = z_vars_im_label[:,:,np.newaxis]
#z_vars_im = np.concatenate((z_vars_im,z_vars_im_label),axis=2)

for i_block in range(i_block_tmp,n_blocks):
    c = 0
    print("Block:",i_block)

    train_tmp = discriminator.model.downsample_to_block(Variable(torch.from_numpy(train).cuda(),requires_grad=False),discriminator.model.cur_block).data.cpu()
    #old_train_tmp = train_tmp[:,0,:,:].view(train_tmp.shape[0],1,train_tmp.shape[2],train_tmp.shape[3])
    #new_train_tmp = train_tmp.clone()
    #train_tmp_fft = fourier_discriminator.model.downsample_to_block(Variable(torch.from_numpy(fft_train).cuda(),requires_grad=False),fourier_discriminator.model.cur_block).data.cpu()
    train_tmp_fft = torch.tensor(np.abs(np.fft.rfft(train_tmp,axis=2))).cuda()#torch.tensor(np.real(np.fft.rfft(train_tmp,axis=2))**2)

    #train_tmp_fft = train_tmp_fft[:,:,:,:]
    #train_tmp_fft = torch.log(train_tmp_fft)
    train_mean = torch.mean(train_tmp,(0,2)).squeeze()
    train_std = torch.sqrt(torch.mean((train_tmp-train_mean)**2,dim=(0,1,2)))
    fft_mean = torch.mean(train_tmp_fft,(0,2)).squeeze().cuda()
    fft_std = torch.sqrt(torch.mean((train_tmp_fft-fft_mean)**2,dim=(0,1,2)))#torch.std(torch.std(train_tmp_fft,0),1).squeeze().cuda()
    #fft_max = torch.max(torch.max(torch.abs(train_tmp_fft),0)[0],1)[0].squeeze().cuda()
    #print("MEAN",fft_mean,"STD",fft_std,"MAX",fft_max)


    for i_epoch in range(i_epoch_tmp,block_epochs[i_block]):
        i_epoch_tmp = 0
        print("Epoch:",i_epoch)
        if fade_alpha<1:
            fade_alpha += 1./rampup
            generator.model.alpha = fade_alpha
            discriminator.model.alpha = fade_alpha
            fourier_discriminator.model.alpha = fade_alpha
            AC_discriminator.model.alpha = fade_alpha
        
        batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)
        print("n_batches: ",len(batches))
        
        anim_idx = np.where(labels_im==1.)
        anim_idx = (anim_idx[0],np.floor(anim_idx[1]/(2**(n_blocks-1-i_block))).astype(np.int))
        anim_labels = np.zeros(shape=(700,int(input_length/(2**(n_blocks-1-i_block)))))
        anim_labels[anim_idx]=1.
        np.save("Animate/labels_block_%i.npy"%i_block,anim_labels)
        if i_epoch%10 == 0:
            #writing for animation
            animate_z_var = Variable(torch.from_numpy(z_vars_im[0,:][np.newaxis,:]),requires_grad=False).cuda()
            animated_signal = generator(animate_z_var).data.detach().cpu().numpy().squeeze()
            np.save("Animate/block_%i_epoch_%i.npy"%(i_block,i_epoch),animated_signal)
        
        #batches = functions.get_batches_new(input_length,n_batch,[0],train)
        iters = int(len(batches)/n_critic)
        for it in range(iters):
            for i_critic in range(n_critic):
                train_batches = train_tmp[batches[it*n_critic+i_critic]]
                #Fixing labels getting downsampled
                idxes = np.nonzero(train_batches[:,:,:,-1])
                idxes = (idxes[:,0],idxes[:,1],idxes[:,2])
                train_batches[:,:,:,-1][idxes] = 1.

                batch_real = Variable(train_batches,requires_grad=True).cuda()
                #batch_real_old = batch_real[:,0,:,:].view(batch_real.shape[0],1,batch_real.shape[2],batch_real.shape[3])

                z_vars = rng.normal(0,1,size=(len(batches[it*n_critic+i_critic]),n_z)).astype(np.float32)
                """
                #Conditional
                z_vars_label = np.zeros(shape=(len(batches[it*n_critic+i_critic]),n_z))
                random_times = np.random.randint(0,n_z,size=(len(batches[it*n_critic+i_critic])))
                z_vars_label[np.arange(len(batches[it*n_critic+i_critic])),random_times] = 1.
                z_vars_label = z_vars_label.astype(np.float32)
                z_vars = z_vars[:,:,np.newaxis]
                z_vars_label = z_vars_label[:,:,np.newaxis]
                z_vars = np.concatenate((z_vars,z_vars_label),axis=2)
                """
                #New_conditional
                #z_vars_label = np.zeros(shape=(len(batches[it*n_critic+i_critic]),input_length))
                if conditional:
                    ##random_times = np.random.randint(0,input_length-80,size=(len(batches[it*n_critic+i_critic]))).astype(np.int)
                    labels_big = np.zeros(shape=(batch_real.shape[0],input_length)).astype(np.float32)
                    #labels_big_new = np.zeros(shape=(batch_real.shape[0],input_length,n_chans)).astype(np.float32)
                    #labels_big_new = np.zeros(shape=(batch_real.shape[0],input_length)).astype(np.float32)
                    #labels_big_new = rng.normal(0,1,size=(batch_real.shape[0],input_length,n_chans)).astype(np.float32)
                    for i in range(len(batches[it*n_critic+i_critic])):
                        n_spikes = int(np.random.normal(spikes_mean,spikes_std))
                        if n_spikes<0:
                            n_spikes=0
                        random_times = np.random.randint(41,input_length-41,size=(n_spikes)).astype(np.int)
                        #random_temps = np.random.randint(0,templates.shape[0],size=(n_spikes)).astype(np.int)
                        for j in range(n_spikes):
                            labels_big[i,random_times[j]:(random_times[j]+label_length)] = 1.
                            #labels_big_new[i,(random_times[j]-41):(random_times[j]+41),:] = templates[random_temps[j],:,:].astype(np.float32)
                            #labels_big_new[i,(random_times[j]-41):(random_times[j]+41)] = templates_new[random_temps[j],:].astype(np.float32)
                    #index = np.where(labels_big==1.)
                    #index = (index[0],np.floor(index[1]/(2**6)).astype(np.int))
                    #labels = np.zeros(shape=(len(batches[it*n_critic+i_critic]),n_z))
                    #labels[index] = 1.
                    labels = labels_big.astype(np.float32)
                    #labels_big = np.mean(labels_big_new,axis=2)
                    z_vars = np.concatenate((z_vars,labels_big),axis=1)


                #z_vars_label[np.arange(len(batches[it*n_critic+i_critic])),random_times] = 1.
                #z_vars_label = z_vars_label.astype(np.float32)
                #z_vars_label = torch.from_numpy(z_vars_label).cuda()

                #test_array = torch.from_numpy(np.ones(shape=(len(batches[it*n_critic+i_critic]),1,256,1)).astype(np.float32)).cuda()
                z_vars = Variable(torch.from_numpy(z_vars),requires_grad=False).cuda()

                batch_fake = Variable(generator(z_vars).data,requires_grad=True).cuda()

                #labels = labels_big_new
                """
                blockreduction = [[32],[16],[8],[4],[2],[]]

                labels = labels[:,np.newaxis,:]
                labels = labels.astype(np.float32)

                labels = torch.from_numpy(labels).cuda()
                for i in range(len(blockreduction[i_block])):
                    labels = torch.nn.AvgPool1d(blockreduction[i_block][i],stride=blockreduction[i_block][i])(labels)
                
                labels = labels[:,:,:,np.newaxis]
                """
                #Conditional
                labels = np.zeros(shape=(batch_fake.shape[0],batch_fake.shape[2]))
                index = np.where(labels_big==1.)
                index = (index[0],np.floor(index[1]/(2**(n_blocks-1-i_block))).astype(np.int))
                labels[index] = 1.

                labels = labels.astype(np.float32)
                labels = labels[:,np.newaxis,:,np.newaxis]
                labels = torch.from_numpy(labels).cuda()
                batch_fake = torch.cat((batch_fake,labels),dim=3)
                #batch_fake = torch.cat((batch_fake,labels),dim=1)

                #batch_fake = torch.cat((batch_fake,labels),dim=3)

                batch_real_fft = torch.transpose(torch.rfft(torch.transpose(batch_real[:,:,:,:],2,3),1,normalized=False),2,3)
                batch_real_fft = torch.sqrt(batch_real_fft[:,:,:,:,0]**2+batch_real_fft[:,:,:,:,1]**2)#batch_real_fft[:,:,:,:,0]**2
                batch_fake_fft = torch.transpose(torch.rfft(torch.transpose(batch_fake[:,:,:,:],2,3),1,normalized=False),2,3)
                batch_fake_fft = torch.sqrt(batch_fake_fft[:,:,:,:,0]**2+batch_fake_fft[:,:,:,:,1]**2)#batch_fake_fft[:,:,:,:,0]**2
                
                #batch_fake_fft = torch.log(batch_fake_fft)
                #batch_real_fft = torch.log(batch_real_fft)

                fake_mean = torch.mean(batch_fake_fft,(0,2)).squeeze()

                #fft_std = torch.sqrt(torch.mean((train_tmp_fft-fft_mean)**2,dim=(0,1,2)))
                fake_std = torch.sqrt(torch.mean((batch_fake_fft-fake_mean)**2,dim=(0,1,2)))
                real_mean = torch.mean(batch_real_fft,(0,2)).squeeze()#fft_mean
                real_std = torch.sqrt(torch.mean((batch_real_fft-real_mean)**2,dim=(0,1,2)))#fft_std
                #NORMALIZING OVER BATCH ONLY
                #fake_mean = torch.mean(batch_fake_fft,(0)).squeeze()
                #fake_std = torch.std(batch_fake_fft,0).squeeze()
                #real_mean = torch.mean(batch_real_fft,(0)).squeeze()
                #real_std = torch.std(batch_real_fft,0).squeeze()

                batch_fake_fft = ((batch_fake_fft-fake_mean)/fake_std)#/fake_max
                batch_real_fft = ((batch_real_fft-real_mean)/real_std)#/real_max

                #batch_fake_fft = torch.mean(batch_fake_fft,dim=0).view(1,batch_fake_fft.shape[1],batch_fake_fft.shape[2],batch_fake_fft.shape[3])
                #batch_real_fft = torch.mean(batch_real_fft,dim=0).view(1,batch_real_fft.shape[1],batch_real_fft.shape[2],batch_real_fft.shape[3])

                #print("FFT-shape",batch_real_fft.shape,"Autocor shape",batch_real_autocor.shape)

                loss_f = fourier_discriminator.train_batch(batch_real_fft,batch_fake_fft)
                
                #AC_discriminator.train_batch(batch_real_autocor,batch_fake_autocor)

                loss_d = discriminator.train_batch(batch_real,batch_fake)

                #print("loss_d",loss_d)
                assert np.all(np.isfinite(loss_d))
            
            for i_gen in range(n_gen):
                
                z_vars = rng.normal(0,1,size=(n_batch,n_z)).astype(np.float32)

                #Conditional
                #z_vars_label = np.zeros(shape=(n_batch,n_z))
                
                if conditional:
                    ##random_times = np.random.randint(0,input_length-80,size=(n_batch)).astype(np.int)
                    labels = np.zeros(shape=(n_batch,input_length))
                    #labels_new = np.zeros(shape=(n_batch,input_length,n_chans))
                    #labels_new = np.zeros(shape=(n_batch,input_length))
                    #labels_new = rng.normal(0,1,size=(n_batch,input_length,n_chans))
                    for i in range(n_batch):
                        n_spikes = int(np.random.normal(spikes_mean,spikes_std))
                        if n_spikes<0:
                            n_spikes=0
                        #Create n_spikes randomly timed spikes
                        random_times = np.random.randint(41,input_length-41,size=(n_spikes)).astype(np.int)
                        #random_temps = np.random.randint(0,templates.shape[0],size=(n_spikes)).astype(np.int)
                        for j in range(n_spikes):
                            labels[i,random_times[j]:(random_times[j]+label_length)] = 1.
                            #labels_new[i,(random_times[j]-41):(random_times[j]+41),:] = templates[random_temps[j],:,:].astype(np.float32)
                            #labels_new[i,(random_times[j]-41):(random_times[j]+41)] = templates_new[random_temps[j],:].astype(np.float32)
                    #index = np.where(labels==1.)
                    #index = (index[0],np.floor(index[1]/(2**6)).astype(np.int))
                    #labels = np.zeros(shape=(n_batch,n_z))
                    #labels[index] = 1.
                    labels = labels.astype(np.float32)
                    #labels = np.mean(labels_new.astype(np.float32),axis=2)
                    z_vars = np.concatenate((z_vars,labels),axis=1)
                    
                    

                #z_vars_label[np.arange(n_batch),random_times] = 1.
                #z_vars_label = z_vars_label.astype(np.float32)
                #z_vars = z_vars[:,:,np.newaxis]
                #z_vars_label = z_vars_label[:,:,np.newaxis]
                #z_vars = np.concatenate((z_vars,z_vars_label),axis=2)

                z_vars = Variable(torch.from_numpy(z_vars),requires_grad=True).cuda()
                loss_g = generator.train_batch(z_vars,discriminator,fourier_discriminator,AC_discriminator,[i_block,n_blocks,i_epoch],labels)

        losses_d.append(loss_d)
        losses_g.append(loss_g)

        if i_epoch%100 == 0:
            generator.eval()
            discriminator.eval()
            fourier_discriminator.eval()

            print('Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f'%(i_epoch,loss_d[0],loss_d[1],loss_d[2],loss_g))
            """
            try:
                os.remove(modelpath+"\\"+modelname%jobid+'_.data')
            except:
                print("Error Removing old data-file")
                pass
            """
            #joblib.dump((i_block_tmp,i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_.data'),compress=True)
            #joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_%d.data'%i_epoch),compress=True)
            #joblib.dump((n_epochs,n_z,n_critic,batch_size,lr),os.path.join(modelpath,modelname%jobid+'_%d.params'%i_epoch),compress=True)
            #train_tmp = old_train_tmp
            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],d=1/(datafreq/np.power(2,n_blocks-1-i_block)))
            train_fft = np.fft.rfft(train_tmp.numpy(),axis=2)
            #Originally mean over channels, but removed
            train_amps = np.abs(train_fft).mean(axis=0).squeeze()#(np.real(train_fft)**2).mean(axis=3).mean(axis=0).squeeze()

            z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False).cuda()
            batch_fake = generator(z_vars)
            #batch_fake = batch_fake[:,0,:,:].view(batch_fake.shape[0],1,batch_fake.shape[2],batch_fake.shape[3])
            #batch_real= batch_real[:,0,:,:].view(batch_real.shape[0],1,batch_real.shape[2],batch_real.shape[3])

            print("Frechet inception distance:",functions.FID(batch_fake[:760,0,:,0].cpu().detach().numpy(),train_tmp[:,0,:,0].numpy()))
            #torch fft
            torch_fake_fft = np.swapaxes(torch.rfft(np.swapaxes(batch_fake.data.cpu(),2,3),1),2,3)
            torch_fake_fft = torch.sqrt(torch_fake_fft[:,:,:,:,0]**2+torch_fake_fft[:,:,:,:,1]**2)#torch_fake_fft[:,:,:,:,0]**2
            
            #Originally mean over channels, but removed
            fake_amps = torch_fake_fft.data.cpu().numpy().mean(axis=0).squeeze()
            #numpy fft
            #fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(),axis=2)
            #fake_amps = np.abs(fake_fft).mean(axis=3).mean(axis=0).squeeze()
            """
            plt.figure()
            plt.plot(freqs_tmp,np.log(fake_amps),label='numpy')
            plt.plot(freqs_tmp,np.log(torch_fake_amps),label='torch')
            plt.show()
            """
            
            for channel_i in range(2):
                plt.figure()
                log_std_fake = np.std(torch_fake_fft.data.cpu().numpy(),axis=0).squeeze()
                log_std_real = np.std(np.abs(train_fft),axis=0).squeeze()
                logmin = np.min(train_amps[:,channel_i])
                logmax = np.max(train_amps[:,channel_i])
                plt.ylim(logmin-np.abs(logmax-logmin)*0.15,logmax+np.abs(logmax-logmin)*0.15)
                plt.plot(freqs_tmp,fake_amps[:,channel_i],label='Fake')
                plt.plot(freqs_tmp,train_amps[:,channel_i],label='Real')
                low_band = fake_amps[:,channel_i]-log_std_fake[:,channel_i]
                print("Minimum low_band",np.min(low_band),"Maximum low_band",np.max(low_band))
                plt.fill_between(freqs_tmp,fake_amps[:,channel_i]-log_std_fake[:,channel_i],fake_amps[:,channel_i]+log_std_fake[:,channel_i],alpha=0.3,label="±std fake")
                plt.fill_between(freqs_tmp,train_amps[:,channel_i]-log_std_real[:,channel_i],train_amps[:,channel_i]+log_std_real[:,channel_i],alpha=0.3,label="±std real")
                plt.title('Frequency Spektrum - Channel %i'%channel_i)
                plt.xlabel('Hz')
                plt.legend()
                #plt.semilogy()
                plt.savefig(os.path.join(outputpath,"Channel_%d"%channel_i+'_fft_%d_%d.png'%(i_block,i_epoch)))
                plt.close()

            """
            graph = make_dot(batch_fake[0].data.cpu().numpy(),params = dict(generator.named_parameters()))
            graph.format = 'png'
            graph.view(filename='digraph',directory='./')
            """
            batch_fake = batch_fake.data.cpu().numpy()
            batch_real = batch_real.data.cpu().numpy()

            #peak_loc_idx = np.where(z_vars_im_label==1)[1]*2**(i_block+1)
            #peak_loc_idx = (np.arange(z_vars_im_label.shape[0]).astype(np.int),peak_loc_idx.astype(np.int),np.zeros(z_vars_im_label.shape[2]).astype(np.int))
            #peak_loc = np.zeros(shape=(batch_fake.shape[0],batch_fake.shape[2],1))
            #peak_loc[peak_loc_idx] = 1.
            #peak_loc = np.zeros(shape=(batch_fake.shape[0],batch_fake.shape[2]))
            #peak_loc_idx = np.floor(random_times_im/(2**(n_blocks-1-i_block))).astype(np.int)
            #peak_loc[(np.arange(batch_fake.shape[0]),peak_loc_idx)] = 1.
            for channel_i in range(2):
                plt.figure(figsize=(45,30))
                for i in range(1,21,2):
                    plt.subplot(20,2,i)
                    plt.plot(batch_fake[i,:,:,channel_i].squeeze())
                    #plt.plot(peak_loc[i,:]*0.05,alpha=0.5)
                    if i==1:
                        plt.title("Fakes")
                    plt.xticks((),())
                    plt.yticks((),())
                    plt.subplot(20,2,i+1)
                    plt.plot(batch_real[i,:,:,channel_i].squeeze())
                    if i==1:
                        plt.title("Reals")
                    plt.xticks((),())
                    plt.yticks((),())
                plt.subplots_adjust(hspace=0)
                plt.savefig(os.path.join(outputpath,'channel_%d'%channel_i+'_fakes_%d_%d.png'%(i_block,i_epoch)))
                plt.close()



            #WELCH GRAPH
            sf = 500
            yf = np.abs(fft(batch_fake.transpose(0,1,3,2)).transpose(0,1,3,2))
            freqs = fftpack.fftfreq(batch_fake.shape[2])*sf
            mask = freqs>=0
            yf = (yf.transpose(2,0,1,3)[mask]).transpose(1,2,0,3)
            freqs = freqs[mask]
            f,Pxx_den = signal.welch(batch_fake.transpose(0,1,3,2),sf,nperseg=input_length)
            f2,Pxx_den2 = signal.welch(batch_real.transpose(0,1,3,2),sf,nperseg=input_length)
            Pxx_den = Pxx_den.transpose(0,1,3,2)
            Pxx_den2 = Pxx_den2.transpose(0,1,3,2)
            yf = yf.mean(axis=0).squeeze()
            Pxx_den = Pxx_den.mean(axis=0).squeeze()
            Pxx_den2 = Pxx_den2.mean(axis=0).squeeze()
            for channel_i in range(2):
                plt.figure()
                plt.title("Welch graph fake vs real channel %d"%channel_i)
                #plt.plot(freqs,yf[:,channel_i]/yf[:,channel_i].sum()*np.diff(f)[0]/np.diff(freqs)[0],alpha=0.5,label="Fourier")
                plt.plot(freqs_tmp,Pxx_den2[:,channel_i]/Pxx_den[:,channel_i].sum(),label=("Real"))
                plt.plot(freqs_tmp,Pxx_den[:,channel_i]/Pxx_den[:,channel_i].sum(),label=("Fake"))
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("PSD [V**2/Hz]")
                plt.semilogy()
                plt.legend()
                plt.savefig(os.path.join(outputpath,'channel_%d'%channel_i+'_Fourier_Welch_%d_%d.png'%(i_block,i_epoch)))          
                plt.close()

            #CHANNEL CORRELATION
            fig,ax = plt.subplots(1,2,figsize=(8,3))
            corr_fake = functions.channel_correlation(batch_fake)
            corr_real = functions.channel_correlation(batch_real)
            corr_real = corr_real[:corr_fake.shape[0],:corr_fake.shape[1]]
            sns.heatmap(
                corr_fake, 
                ax=ax[0],
                vmin=-1, vmax=1, center=0.5,
                cmap=sns.diverging_palette(20, 220, n=200),
                square=True,
                cbar=False
            )
            sns.heatmap(
                corr_real, 
                ax=ax[1],
                vmin=-1, vmax=1, center=0.5,
                cmap=sns.diverging_palette(20, 220, n=200),
                square=True
            )
            ax[0].title.set_text('Fake')
            ax[1].title.set_text('Real')
            plt.savefig(os.path.join(outputpath,'Correlation_matrix'+'_Block_%d_epoch_%d.png'%(i_block,i_epoch)))          
            plt.close()
            """
            
            plt.figure(figsize=(10,10))
            for i in range(10):
                plt.subplot(10,1,i+1)
                plt.plot(batch_real[i].squeeze())
                plt.xticks((),())
                plt.yticks((),())
            plt.subplots_adjust(hspace=0)
            plt.savefig(os.path.join(outputpath,modelname%jobid+'_reals_%d_%d.png'%(i_block,i_epoch)))
            plt.close()
            """
            """
            try:
                os.remove(modelpath+"\\"+modelname%jobid+'.disc')
                os.remove(modelpath+"\\"+modelname%jobid+'.gen')
                print("models should have been removed now")
            except:
                print("Error removing old disc and/or gen")
                pass
            """

            #torch.save((generator.state_dict(),generator.optimizer.state_dict(),generator.did_init_train),os.path.join(modelpath,modelname%jobid+'.gen'))
            #torch.save((discriminator.state_dict(),discriminator.optimizer.state_dict(),discriminator.did_init_train),os.path.join(modelpath,modelname%jobid+'.disc'))

            #discriminator.save_model(os.path.join(modelpath,modelname%jobid+'.disc'))
            generator.save_model(os.path.join(modelpath,modelname%jobid+'.gen'))
            joblib.dump((i_block,fade_alpha),os.path.join(modelpath,modelname%jobid+'.data'),compress=True)

            plt.figure(figsize=(10,15))
            plt.subplot(3,2,1)
            plt.plot(np.asarray(losses_d)[:,0],label='Loss Real')
            plt.plot(np.asarray(losses_d)[:,1],label='Loss Fake')
            plt.title('Losses Discriminator')
            plt.legend()
            plt.subplot(3,2,2)
            plt.plot(np.asarray(losses_d)[:,0]+np.asarray(losses_d)[:,1]+np.asarray(losses_d)[:,2],label='Loss')
            plt.title('Loss Discriminator')
            plt.legend()
            plt.subplot(3,2,3)
            plt.plot(np.asarray(losses_d)[:,2],label='Penalty Loss')
            plt.title('Penalty')
            plt.legend()
            plt.subplot(3,2,4)
            plt.plot(-np.asarray(losses_d)[:,0]-np.asarray(losses_d)[:,1],label='Wasserstein Distance')
            plt.title('Wasserstein Distance')
            plt.legend()
            plt.subplot(3,2,5)
            plt.plot(np.asarray(losses_g),label='Loss Generator')
            plt.title('Loss Generator')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outputpath,modelname%jobid+'_losses.png'))
            plt.close()

            generator.train()
            discriminator.train()
            fourier_discriminator.train()
            #train_tmp = new_train_tmp


    fade_alpha = 0.
    generator.model.cur_block += 1
    discriminator.model.cur_block -= 1
    fourier_discriminator.model.cur_block -=1
    AC_discriminator.model.cur_block -=1
