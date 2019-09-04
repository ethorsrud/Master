import torch
from torch.autograd import Variable
import torch.nn.functional as F

class VAE_new(torch.nn.Module):
    def __init__(self,input_size,is_training=True):
        super().__init__()
        self.is_training = True
        self.input_size = input_size
        #Input 100x100 -> 10 channels of 100x100
        self.conv1 = torch.nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)

        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.conv3 = torch.nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        #self.bn3 = torch.nn.BatchNorm2d(32)

        self.conv4 = torch.nn.Conv2d(64,16,kernel_size=3,stride=2,padding=1)
        self.bn4 = torch.nn.BatchNorm2d(16)

        self.fc1 = torch.nn.Linear(25 * 25 * 16, 2048)
        self.fc_bn1 = torch.nn.BatchNorm1d(2048)

        self.my_lin = torch.nn.Linear(2048,2048)
        self.sigma_lin = torch.nn.Linear(2048,2048)

        self.fc3 = torch.nn.Linear(2048, 2048)
        self.fc_bn3 = torch.nn.BatchNorm1d(2048)
        self.fc4 = torch.nn.Linear(2048, 25 * 25 * 16)
        self.fc_bn4 = torch.nn.BatchNorm1d(25 * 25 * 16)


        self.conv5 = torch.nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.conv6 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = torch.nn.BatchNorm2d(32)
        self.conv7 = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = torch.nn.BatchNorm2d(16)
        self.conv8 = torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self,input):
        if self.is_training:
            #ENCODE
            x = F.relu(self.bn1(self.conv1(input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))

            x = x.view(-1,16*25*25)

            x = F.relu(self.fc_bn1(self.fc1(x)))

            my = self.my_lin(x)
            sigma = self.sigma_lin(x)
            #MIDDLE
            z = my+torch.exp(sigma*0.5)*torch.randn_like(sigma)#torch.normal(torch.zeros(my.shape[0],my.shape[1])).cuda()
            #z = my+sigma*torch.randn_like(sigma)
            #DECODE
            fc3 = F.relu(self.fc_bn3(self.fc3(z)))
            fc4 = F.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 25, 25)
            conv5 = F.relu(self.bn5(self.conv5(fc4)))
            conv6 = F.relu(self.bn6(self.conv6(conv5)))
            conv7 = F.relu(self.bn7(self.conv7(conv6)))
            #B/W
            #z = self.conv8(conv7).view(-1, 1, 100, 100)
            z = self.conv8(conv7).view(-1, 1, 100, 100)
            
        if not self.is_training:
            #DECODE
            my = torch.tensor([0])
            sigma = torch.tensor([0])
            fc3 = F.relu(self.fc_bn3(self.fc3(input)))
            fc4 = F.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 25, 25)
            conv5 = F.relu(self.bn5(self.conv5(fc4)))
            conv6 = F.relu(self.bn6(self.conv6(conv5)))
            conv7 = F.relu(self.bn7(self.conv7(conv6)))
            #B/W
            #z = self.conv8(conv7).view(-1, 1, 100, 100)
            z = self.conv8(conv7).view(-1, 1, 100, 100)

        return z,my,sigma



        