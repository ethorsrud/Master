#%load_ext autoreload
#%autoreload 2
import os
#import joblib
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
from  datetime import datetime
#from torchviz import make_dot
from my_utils import functions
from scipy import signal
from scipy.fftpack import fft
from scipy import fftpack
import seaborn as sns

#plt.switch_backend('agg')
#Error tracebacking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

torch.cuda.set_device(3)

n_critic = 1#5
n_gen = 1
n_batch = 64#56#64
input_length = 1536#768
jobid = 0

n_z = 200
lr = 0.001#0.001
n_blocks = 6
rampup = 400.#2000.
block_epochs = [400,800,800,800,800,800]#[2000,4000,4000,4000,4000,4000]

task_ind = 0#subj_ind

np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)

datafreq = 500#250#128 #hz
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

train = train-np.mean(train,axis=(0,2)).squeeze()#-train.mean()
train = train/np.std(train,axis=(0,2)).squeeze()#train.std()
train = train/np.max(np.abs(train),axis=(0,2)).squeeze()#np.abs(train).max()


fft_train = np.real(np.fft.rfft(train,axis=2))**2#np.abs(np.fft.rfft(train,axis=2))
#fft_train = np.log(fft_train)
#fft_mean = fft_train.mean()
#fft_std = fft_train.std()
#fft_max = np.abs(fft_train).max()


modelpath = os.path.normpath(other_path+os.sep+"Models"+os.sep+"GAN")
outputpath = os.path.normpath(other_path+os.sep+"Output"+os.sep+"GAN")
modelname = 'Progressive%s'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

generator = Generator(n_chans,n_z)
discriminator = Discriminator(n_chans)
fourier_discriminator = Fourier_Discriminator(n_chans)
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

generator.train()
discriminator.train()
fourier_discriminator.train()
AC_discriminator.train()

losses_d = []
losses_g = []

losses_fourier = []

i_epoch = 0
z_vars_im = rng.normal(0,1,size=(1000,n_z)).astype(np.float32)

for i_block in range(i_block_tmp,n_blocks):
    c = 0
    print("Block:",i_block)

    train_tmp = discriminator.model.downsample_to_block(Variable(torch.from_numpy(train).cuda(),requires_grad=False),discriminator.model.cur_block).data.cpu()
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
        #batches = functions.get_batches_new(input_length,n_batch,[0],train)
        iters = int(len(batches)/n_critic)
        for it in range(iters):
            for i_critic in range(n_critic):
                train_batches = train_tmp[batches[it*n_critic+i_critic]]
                batch_real = Variable(train_batches,requires_grad=True).cuda()
                z_vars = rng.normal(0,1,size=(len(batches[it*n_critic+i_critic]),n_z)).astype(np.float32)
                z_vars = Variable(torch.from_numpy(z_vars),requires_grad=False).cuda()
                batch_fake = Variable(generator(z_vars).data,requires_grad=True).cuda()

                #batch_fake_for_investigation = batch_fake.data.cpu().numpy()
                """
                z_vars_for_investigation = z_vars.cpu().numpy()
                if not np.all(np.isfinite(batch_fake_for_investigation)):
                    print("All z_vars finite?",np.all(np.isfinite(z_vars_for_investigation)))
                """

                batch_real_fft = torch.transpose(torch.rfft(torch.transpose(batch_real,2,3),1,normalized=False),2,3)
                batch_real_fft = torch.sqrt(batch_real_fft[:,:,1:,:,0]**2+batch_real_fft[:,:,1:,:,1]**2)#batch_real_fft[:,:,:,:,0]**2
                batch_fake_fft = torch.transpose(torch.rfft(torch.transpose(batch_fake,2,3),1,normalized=False),2,3)
                batch_fake_fft = torch.sqrt(batch_fake_fft[:,:,1:,:,0]**2+batch_fake_fft[:,:,1:,:,1]**2)#batch_fake_fft[:,:,:,:,0]**2
                
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
                """
                plt.plot(batch_real_fft[0,0,:,0].cpu().detach().numpy(),label="real")
                plt.plot(batch_fake_fft[0,0,:,0].cpu().detach().numpy(),label="fake")
                plt.legend()
                plt.show()
                """
                """
                plt.figure()
                plt.plot(batch_fake_fft[0,0,:,0].cpu().detach().numpy())
                plt.plot(batch_real_fft[0,0,:,0].cpu().detach().numpy())
                plt.legend(["Fake","Real"])
                plt.savefig(os.path.join(outputpath+"test_epoch_%i_it_%i_icrit_%i.png"%(i_epoch,it,i_critic)))
                plt.close()
                """
                
                #print("MIN(Fake): ",torch.min(batch_fake_fft),"MIN(Real)",torch.min(batch_real_fft))
                #print("MAX(Fake): ",torch.max(batch_fake_fft),"MAX(Real)",torch.max(batch_real_fft))
                #batch_real_autocor = functions.autocorrelation(batch_real)
                #batch_fake_autocor = functions.autocorrelation(batch_fake)

                #print("FFT-shape",batch_real_fft.shape,"Autocor shape",batch_real_autocor.shape)

                fourier_discriminator.train_batch(batch_real_fft,batch_fake_fft)
                #AC_discriminator.train_batch(batch_real_autocor,batch_fake_autocor)
                loss_d = discriminator.train_batch(batch_real,batch_fake)
                assert np.all(np.isfinite(loss_d))
            
            for i_gen in range(n_gen):
                z_vars = rng.normal(0,1,size=(n_batch,n_z)).astype(np.float32)
                z_vars = Variable(torch.from_numpy(z_vars),requires_grad=True).cuda()
                loss_g = generator.train_batch(z_vars,discriminator,fourier_discriminator,AC_discriminator)

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
            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],d=1/(datafreq/np.power(2,n_blocks-1-i_block)))
            train_fft = np.fft.rfft(train_tmp.numpy(),axis=2)

            #Originally mean over channels, but removed
            train_amps = np.abs(train_fft).mean(axis=0).squeeze()#(np.real(train_fft)**2).mean(axis=3).mean(axis=0).squeeze()

            z_vars = Variable(torch.from_numpy(z_vars_im),requires_grad=False).cuda()
            batch_fake = generator(z_vars)

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
                log_std_real = np.std(train_fft,axis=0).squeeze()
                logmin = np.min(np.log(train_amps[:,channel_i]))
                logmax = np.max(train_amps[:,channel_i])
                plt.ylim(logmin-np.abs(logmax-logmin)*0.15,logmax+np.abs(logmax-logmin)*0.15)
                plt.plot(freqs_tmp,fake_amps[:,channel_i],label='Fake')
                plt.plot(freqs_tmp,train_amps[:,channel_i],label='Real')
                plt.fill_between(freqs_tmp,fake_amps[:,channel_i]-log_std_fake[:,channel_i],fake_amps[:,channel_i]+log_std_fake[:,channel_i],alpha=0.3,label="±std fake")
                plt.fill_between(freqs_tmp,train_amps[:,channel_i]-log_std_real[:,channel_i],train_amps[:,channel_i]+log_std_real[:,channel_i],alpha=0.3,label="±std real")
                plt.title('Frequency Spektrum - Channel %i'%channel_i)
                plt.xlabel('Hz')
                plt.legend()
                plt.semilogy()
                plt.savefig(os.path.join(outputpath,"Channel_%d"%channel_i+'_fft_%d_%d.png'%(i_block,i_epoch)))
                plt.close()

            """
            graph = make_dot(batch_fake[0].data.cpu().numpy(),params = dict(generator.named_parameters()))
            graph.format = 'png'
            graph.view(filename='digraph',directory='./')
            """
            batch_fake = batch_fake.data.cpu().numpy()
            batch_real = batch_real.data.cpu().numpy()

            for channel_i in range(2):
                plt.figure(figsize=(20,10))
                for i in range(1,21,2):
                    plt.subplot(20,2,i)
                    plt.plot(batch_fake[i,:,:,channel_i].squeeze())
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
            sns.heatmap(
                corr_fake, 
                ax=ax[0],
                vmin=0, vmax=1, center=0.5,
                cmap=sns.diverging_palette(20, 220, n=200),
                square=True,
                cbar=False
            )
            sns.heatmap(
                corr_real, 
                ax=ax[1],
                vmin=0, vmax=1, center=0.5,
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
            #generator.save_model(os.path.join(modelpath,modelname%jobid+'.gen'))

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


    fade_alpha = 0.
    generator.model.cur_block += 1
    discriminator.model.cur_block -= 1
    fourier_discriminator.model.cur_block -=1
    AC_discriminator.model.cur_block -=1
