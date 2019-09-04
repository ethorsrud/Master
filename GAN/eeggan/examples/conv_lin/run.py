#%load_ext autoreload
#%autoreload 2
import os
import joblib
import sys
import mne
#sys.path.append("/home/hartmank/git/GAN_clean")
#sys.path.append("/Users/eirikthorsrud/desktop/Master/GAN-master")
sys.path.append("C:\\Users\\eiri-\\Documents\\github\\Master\\GAN")
from braindecode.datautil.iterators import get_balanced_batches
from eeggan.examples.conv_lin.model import Generator,Discriminator
from eeggan.util import weight_filler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
from  datetime import datetime
#Just for saving files
now = datetime.now()

plt.switch_backend('agg')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

n_critic = 5
#So that n_batch*n_critic >= number of samples
n_batch = 56#64
input_length = 768#768
jobid = 0

n_z = 200
lr = 0.001
n_blocks = 6
rampup = 2000.
block_epochs = [2000,4000,4000,4000,4000,4000]

#subj_ind = int(os.getenv('SLURM_ARRAY_TASK_ID','0'))
task_ind = 0#subj_ind
#subj_ind = 9
subj_names = ['BhNoMoSc1',
             'FaMaMoSc1',
             'FrThMoSc1',
             'GuJoMoSc01',
             'KaUsMoSc1',
             'LaKaMoSc1',
             'LuFiMoSc3',
             'MaJaMoSc1',
             'MaKiMoSC01',
             'MaVoMoSc1',
             'PiWiMoSc1',
             'RoBeMoSc03',
             'RoScMoSc1',
             'StHeMoSc01']

np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)

#data = os.path.join('/data/schirrmr/hartmank/data/GAN/cnt',subj_names[subj_ind]+'_FCC4h.cnt')
#data = os.path.join('C:\\Users\\eiri-\\OneDrive\\Skrivebord\\Master_Windows\\Dataset\\BCICIV_2a_gdf\\A01T.gdf')
data = os.path.join('C:\\Users\\eiri-\\Documents\\github\\Dataset\\dataset_BCIcomp1.mat')
#data = os.path.join('C:\\Users\\eiri-\\OneDrive\\Skrivebord\\Master_Windows\\Dataset\\sp1s_aa_1000hz.mat')
#data = os.path.join('C:\\Users\\eiri-\\OneDrive\\Skrivebord\\Master_Windows\\Dataset\\BCICIV_1_mat\\BCICIV_calib_ds1a.mat')
#EEG_data = joblib.load(data)
#EEG_data = mne.io.read_raw_gdf(data,preload=True)
EEG_data = scipy.io.loadmat(data)
#train_set = EEG_data['train_set']
#test_set = EEG_data['test_set']
#train = np.concatenate((train_set.X,test_set.X))
train = EEG_data['x_train']
test = EEG_data['x_test']
target = EEG_data['y_train']
test = test[:768,0,:,None][:,np.newaxis,:,:]
test = np.swapaxes(test,0,2).astype(np.float32)
#target = np.concatenate((train_set.y,test_set.y))
#print("Test shape",test.shape)
train = train[:,:,:,None]
train = train-train.mean()
train = train/train.std()
train = train/np.abs(train).max()
#target_onehot = np.zeros((target.shape[0],2))
#target_onehot = np.zeros((train.shape[0],2))
#target_onehot[:,target] = 1
#print(train.shape)
train = np.swapaxes(train,0,2)
train = np.swapaxes(train,1,2)
#train = train[:,:,:,:]
train = np.swapaxes(train,1,2)
train = train[:,0,:768,:][:,np.newaxis,:,:].astype(np.float32)
#train = np.ones((140,1,768,1)).astype(np.float32)
train = np.concatenate((train,test))
#print(train.shape)
datafreq = 128 #hz

#modelpath = '/data/schirrmr/hartmank/data/GAN/models/GAN_debug/%s/'%('PAPERFIN4_'+subj_names[subj_ind]+'_FFC4h_WGAN_adaptlambclamp_CONV_LIN_10l_run%d'%task_ind)
modelpath = 'C:\\Users\\eiri-\\Documents\\github\\Models\\GAN'
outputpath = 'C:\\Users\\eiri-\\Documents\\github\\Output\\GAN'
modelname = 'Progressive%s'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

generator = Generator(1,n_z)
discriminator = Discriminator(1)

generator.train_init(alpha=lr,betas=(0.,0.99))
discriminator.train_init(alpha=lr,betas=(0.,0.99),eps_center=0.001,
                        one_sided_penalty=True,distance_weighting=True)
generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)

i_block_tmp = 0
i_epoch_tmp = 0
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = n_blocks-1-i_block_tmp
fade_alpha = 1.
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha

generator = generator.cuda()
discriminator = discriminator.cuda()
generator.train()
discriminator.train()


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


losses_d = []
losses_g = []
i_epoch = 0
z_vars_im = rng.normal(0,1,size=(1000,n_z)).astype(np.float32)

for i_block in range(i_block_tmp,n_blocks):
    c = 0
    print("Block:",i_block)

    train_tmp = discriminator.model.downsample_to_block(Variable(torch.from_numpy(train).cuda(),volatile=True),discriminator.model.cur_block).data.cpu()

    for i_epoch in range(i_epoch_tmp,block_epochs[i_block]):
        i_epoch_tmp = 0
        print("Epoch:",i_epoch)
        if fade_alpha<1:
            fade_alpha += 1./rampup
            generator.model.alpha = fade_alpha
            discriminator.model.alpha = fade_alpha
        
        batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)
        iters = int(len(batches)/n_critic)
        for it in range(iters):
            for i_critic in range(n_critic):
                train_batches = train_tmp[batches[it*n_critic+i_critic]]
                batch_real = Variable(train_batches,requires_grad=True).cuda()
                z_vars = rng.normal(0,1,size=(len(batches[it*n_critic+i_critic]),n_z)).astype(np.float32)
                z_vars = Variable(torch.from_numpy(z_vars),volatile=True).cuda()
                batch_fake = Variable(generator(z_vars).data,requires_grad=True).cuda()

                loss_d = discriminator.train_batch(batch_real,batch_fake)
                assert np.all(np.isfinite(loss_d))
            z_vars = rng.normal(0,1,size=(n_batch,n_z)).astype(np.float32)
            z_vars = Variable(torch.from_numpy(z_vars),requires_grad=True).cuda()
            loss_g = generator.train_batch(z_vars,discriminator)

        losses_d.append(loss_d)
        losses_g.append(loss_g)


        if i_epoch%100 == 0:
            generator.eval()
            discriminator.eval()

            print('Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f'%(i_epoch,loss_d[0],loss_d[1],loss_d[2],loss_g))
            """
            try:
                os.remove(modelpath+"\\"+modelname%jobid+'_.data')
            except:
                print("Error Removing old data-file")
                pass
            """
            joblib.dump((i_block_tmp,i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_.data'),compress=True)
            #joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_%d.data'%i_epoch),compress=True)
            #joblib.dump((n_epochs,n_z,n_critic,batch_size,lr),os.path.join(modelpath,modelname%jobid+'_%d.params'%i_epoch),compress=True)

            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],d=1/(datafreq/np.power(2,n_blocks-1-i_block)))

            train_fft = np.fft.rfft(train_tmp.numpy(),axis=2)
            train_amps = np.abs(train_fft).mean(axis=3).mean(axis=0).squeeze()


            z_vars = Variable(torch.from_numpy(z_vars_im),volatile=True).cuda()
            batch_fake = generator(z_vars)
            fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(),axis=2)
            fake_amps = np.abs(fake_fft).mean(axis=3).mean(axis=0).squeeze()

            plt.figure()
            plt.plot(freqs_tmp,np.log(fake_amps),label='Fake')
            plt.plot(freqs_tmp,np.log(train_amps),label='Real')
            plt.title('Frequency Spektrum')
            plt.xlabel('Hz')
            plt.legend()
            plt.savefig(os.path.join(outputpath,modelname%jobid+'_fft_%d_%d.png'%(i_block,i_epoch)))
            plt.close()

            batch_fake = batch_fake.data.cpu().numpy()
            plt.figure(figsize=(10,10))
            for i in range(10):
                plt.subplot(10,1,i+1)
                plt.plot(batch_fake[i].squeeze())
                plt.xticks((),())
                plt.yticks((),())
            plt.subplots_adjust(hspace=0)
            plt.savefig(os.path.join(outputpath,modelname%jobid+'_fakes_%d_%d.png'%(i_block,i_epoch)))
            plt.close()
            """
            try:
                os.remove(modelpath+"\\"+modelname%jobid+'.disc')
                os.remove(modelpath+"\\"+modelname%jobid+'.gen')
                print("models should have been removed now")
            except:
                print("Error removing old disc and/or gen")
                pass
            """

            torch.save((generator.state_dict(),generator.optimizer.state_dict(),generator.did_init_train),os.path.join(modelpath,modelname%jobid+'.gen'))
            torch.save((discriminator.state_dict(),discriminator.optimizer.state_dict(),discriminator.did_init_train),os.path.join(modelpath,modelname%jobid+'.disc'))

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


    fade_alpha = 0.
    generator.model.cur_block += 1
    discriminator.model.cur_block -= 1
