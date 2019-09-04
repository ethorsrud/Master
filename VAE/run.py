import numpy as np
from VAE import VAE
import os
import cv2
import torch
import matplotlib.pyplot as plt
import imageio
import torch.nn.functional as F
from VAE_dnn import VAE_dnn
from VAE_new import VAE_new
import joblib


torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
MODELDIR = 'C:\\Users\\eiri-\\Documents\\github\\Models\\VAE'
DATADIR = 'C:\\Users\\eiri-\\Documents\\github\\Dataset\\PetImages'
OUTPUTDIR = 'C:\\Users\\eiri-\\Documents\\github\\Output\\VAE'

if not os.path.exists(MODELDIR):
    os.makedirs(MODELDIR)

if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

CATEGORIES = ["one_dog"]#["5000_dogs"]#["small_dog"]

IMG_SIZE = 100
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #B/W
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                #img_array = cv2.imread(os.path.join(path,img))
                #B/W
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                #new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                #new_array = np.swapaxes(new_array,1,2)
                #new_array = np.swapaxes(new_array,0,1)
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()
train_images = []
y = []
for features,label in training_data:
    train_images.append(features)
    y.append(label)

train_images = np.array(train_images)


#B/W
train_images = train_images.reshape(train_images.shape[0],1,IMG_SIZE,IMG_SIZE).astype('float32')
#train_images = train_images.reshape(train_images.shape[0],3,IMG_SIZE,IMG_SIZE).astype('float32')

train_images = (train_images-127.5)/127.5

train_images = torch.from_numpy(train_images).cuda()
batch_size= 50
tmp_epoch = 0
#autoencoder = VAE(100)
#autoencoder = VAE_dnn(100)
autoencoder = VAE_new(100)
autoencoder = autoencoder.cuda()

#lossfunction = torch.nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(),lr = 1e-3)

#LOAD
try:
    autoencoder.load_state_dict(torch.load(os.path.join(MODELDIR,"model.file")))
    autoencoder.eval()
    tmp_epoch = joblib.load(os.path.join(MODELDIR,'current_epoch.file'))
    print("Model lock'n'loaded")

except:
    print("No model found, creating new")
    pass

def calculate_loss(x, recon_x, mean, log_var):
    # reconstruction loss
    x = x.reshape(-1,IMG_SIZE*IMG_SIZE)
    recon_x = recon_x.reshape(-1,IMG_SIZE*IMG_SIZE)
    #x = x.reshape(-1,IMG_SIZE*IMG_SIZE*3)
    #recon_x = recon_x.reshape(-1,IMG_SIZE*IMG_SIZE*3)
    #BCE = torch.nn.BCELoss(reduction='sum')
    #BCE = BCE(recon_x,x)
    #BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    recon_loss = torch.nn.MSELoss(reduction="sum")
    #recon_loss = x*torch.log(1e-10+recon_x)+(1-x)*torch.log(1e-10+1-recon_x)
    #recon_loss = -torch.sum(recon_loss,axis=1)
    #recon_loss = torch.mean(recon_loss)


    MSE = recon_loss(recon_x,x)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    #print(KLD)
    #KLD = 0.5 * torch.sum(-1 + log_var.pow(2)+ mean.pow(2) - log_var.pow(2).log())
    #KLD = torch.nn.KLDivLoss(size_average=False)
    #KLD = KLD(recon_x,x)
    return MSE+KLD

autoencoder.train()
for epoch in range(tmp_epoch+1,100000):
    
    permutation = torch.randperm(train_images.shape[0])
    for i  in range(0,train_images.shape[0],batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch = train_images[indices]
        out_images,my,sigma = autoencoder(batch)
        #B/W
        out_images = out_images.reshape(-1,1,IMG_SIZE,IMG_SIZE)
        #out_images = out_images.reshape(-1,3,IMG_SIZE,IMG_SIZE)
        loss = calculate_loss(batch,out_images,my,sigma)
        #loss = lossfunction(out_images,train_images)
        loss.backward()
        optimizer.step()
    print("After epoch %i, loss = %f"%(epoch,loss.item()))

    if epoch%10==0:
        autoencoder.eval()
        random_image = torch.rand_like(torch.zeros(1,2048)).cuda()
        autoencoder.is_training = False
        random_out,my,sigma = autoencoder(random_image)
        #B/W
        random_out = random_out.reshape(-1,1,IMG_SIZE,IMG_SIZE)
        #random_out = random_out.reshape(-1,3,IMG_SIZE,IMG_SIZE)
        autoencoder.is_training = True
        #B/W
        plt.imsave(os.path.join(OUTPUTDIR,"epoch_%i.png"%epoch) , random_out[0,0, :, :].cpu().detach(), cmap='gray')
        #img = (random_out[0,:, :, :]*127.5+127.5).cpu().detach().numpy().astype("uint8")
        #img = np.swapaxes(img,0,1)
        #img = np.swapaxes(img,1,2)

        #plt.imsave("epoch_%i.png"%epoch,img)

        joblib.dump(epoch,os.path.join(MODELDIR,'current_epoch.file'),compress=True)
        torch.save(autoencoder.state_dict(),os.path.join(MODELDIR,"model.file"))
        autoencoder.train()
            
        


