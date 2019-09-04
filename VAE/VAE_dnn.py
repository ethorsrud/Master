import torch
from torch.autograd import Variable
import torch.nn.functional as F

class VAE_dnn(torch.nn.Module):
    def __init__(self,input_size,is_training=True):
        super().__init__()
        self.is_training = True
        self.input_size = input_size
        self.dl1 = torch.nn.Linear(input_size*input_size,1000)
        self.dl2 = torch.nn.Linear(1000,200)
        #Input 100x100 -> 10 channels of 100x100
        #self.conv1 = torch.nn.Conv2d(1,10,kernel_size=3,stride=1,padding=1)
        #pool reduces the size with a factor of 2
        #self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        #Reduces 20 channels to 10
        #self.conv2 = torch.nn.Conv2d(10,10,kernel_size=3,stride=1,padding=1)
        
        self.my_lin = torch.nn.Linear(200,20)
        self.sigma_lin = torch.nn.Linear(200,20)

        self.Upsample1 = torch.nn.Linear(20,200)
        self.Upsample2 = torch.nn.Linear(200,1000)
        self.Upsample3 = torch.nn.Linear(1000,10000)

    def forward(self,input):
        if self.is_training:
            #ENCODE
            input = input.view(-1,input.shape[2]*input.shape[3])
            x = F.relu(self.dl1(input))
            x = F.relu(self.dl2(x))

            my = self.my_lin(x)
            sigma = self.sigma_lin(x)
            #MIDDLE
            z = my+torch.exp(0.5*sigma)*torch.randn_like(sigma)#torch.normal(torch.zeros(my.shape[0],my.shape[1])).cuda()
            #DECODE
            z = F.relu(self.Upsample1(z))
            z = F.relu(self.Upsample2(z))
            z = F.sigmoid(self.Upsample3(z))
            
            z = z.view(-1,self.input_size,self.input_size)
        if not self.is_training:
            #DECODE
            my = torch.tensor([0])
            sigma = torch.tensor([0])
            z = F.relu(self.Upsample1(input))
            z = F.relu(self.Upsample2(z))
            z = F.sigmoid(self.Upsample3(z))
            
            z = z.view(-1,self.input_size,self.input_size)

        return z,my,sigma



        