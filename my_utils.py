import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch

"""
Just a file including self-written functions needed for the project
"""


class functions():
    def get_balanced_batches(n_samples,batch_size):
        """
        Returns a list of length n_batches arrays with indexes of samples
        
        example:
        3 batches, batch_size = 3
        [array([1,2,3]),array([1,3,4]),array([5,2,1])]


        NEW VERSION below is faster and more compatible

        """
        main_list = []
        samples = []
        for i in range(n_samples):
            samples.append(int(i))
        random.shuffle(samples)
        n_batches = int(n_samples/batch_size)
        for i in range(n_batches):
            tmp_list = []
            for j in range(batch_size):
                tmp_list.append(samples[0])
                del samples[0]
                
            main_list.append(np.array(tmp_list))
        return main_list
    
    def get_batches_new(split,n_batches,channels,data):
        """
        Randomly slices up a signal of a given length channelwise

        inputs: 
            split: Length of each sample
            n_batches: Number of wanted batches
            channels: List of channels you want random batches from
            data: Dataset with shape [n_samples,channels]

        returns:
            batches of small random samples from one long sample
            with shape: [n_batches,channels,data,1]

        """
        data_len = data.shape[0]
        max_int = data_len-split
        #Only the selected channels
        #ch_data = 
        batches = [] 
        for i in range(len(channels)):
            random_ints = np.random.randint(0,max_int,size=(n_batches,1))
            batches.append(data[:,channels][:,i][random_ints+np.arange(split)])
        return np.swapaxes(np.array(batches),0,1)[:,:,:,np.newaxis]
    
    def check_batch(batches,data,split):
        """
        Just a validation check to see if
        function get_batches_new works as intended

        Function: Searches for the splitted sample in the original data
        """
        #Second batch, channel 1
        sample = batches[1,1,:,0]
        #original channel 1
        ch = data[:,1]
        for i in range(ch.shape[0]-split):
            slice = ch[i:(i+split)]
            MSE = np.mean((sample-slice)**2)
            if MSE==0:
                print("BATCH FOUND")
                break
        return 0

    def samp_from_freq(n_samples):
        """
        An attempt of sampling a signal from a self made frequency spectrum
        """
        #Generating frequency spectrum
        x = np.linspace(0,100,251)
        x2 = np.linspace(0,5,251)
        spectrum = 50*np.exp(-(x-30)**2/2)
        spectrum += 60*np.sin(np.random.randn(251)*2*np.pi)*np.exp(-x2)
        #spectrum += np.random.randn(251)
        plt.plot(spectrum)
        plt.show()
        signal = np.fft.irfft(spectrum)
        plt.plot(signal)
        plt.show()

        return signal
    
    def autocorrelation(signals):
        """
        input: signals - Signals in shape [n_signals,1,time_samples,channels]

        Output: autocorrelated signals with themself in shape [n_signals,1,time_samples/2,channels]
        """

        n = signals.shape[2]
        M = int(n/2)

        means = torch.mean(signals,dim=2)
        stds = torch.std(signals,dim=2)
        #Centering
        signals = (signals.permute(2,0,1,3)-means).permute(1,2,0,3)

        if signals.is_cuda:
            C = torch.zeros(size=(signals.shape[0],signals.shape[1],M,signals.shape[3])).cuda()
        else:
            C = torch.zeros(size=(signals.shape[0],signals.shape[1],M,signals.shape[3]))

        for i in range(M):
            C[:,:,i,:] = torch.sum(signals[:,:,:M,:]*signals[:,:,i:(M+i),:],dim=2)/torch.sum(signals[:,:,:M,:]*signals[:,:,:M,:],dim=2)
            C[:,:,i,:] *= (1./(M*stds**2))
        print(C.shape)
        C = ((C.permute(2,0,1,3)-torch.mean(C,dim=2))/torch.std(C,dim=2)).permute(1,2,0,3)
        print(C.shape)
        return C


"""
#Testing autocorrelation
x = torch.from_numpy(np.linspace(0,6*np.pi,500))
signal1 = torch.stack([torch.sin(x)+1.0,torch.sin(x)+2.0])
signal2 = torch.stack([torch.cos(x)+3.0,torch.cos(x)+4.0])
signals = torch.stack([signal1,signal2])
signals = signals[:,np.newaxis,:,:]
signals = torch.transpose(signals,2,3)
autocor = functions.autocorrelation(signals)
plt.plot(x[:250],autocor[0,0,:,0])
plt.show()
"""



"""
#Just some testing
lr = 100
n=1000
colors = plt.cm.rainbow(np.linspace(0,1,n))
x = np.linspace(0,10,500)
signal1 = np.sin(x)
signal2 = np.cos(x)
signals = np.array([signal1])
signals = torch.from_numpy(signals)
signals.requires_grad=True
autocor = functions.autocorrelation(signals)
mean = torch.mean(autocor)
mean.backward()
signals = (signals+signals.grad*lr).detach()
for i in range(n):
    signals.requires_grad=True
    autocor = functions.autocorrelation(signals)
    plt.plot(x[:250],autocor[0].detach(),color=colors[i])
    mean = torch.mean(autocor)
    print(i)
    print(mean)
    mean.backward()
    print(signals.grad)
    signals = (signals+signals.grad*lr).detach()
    
plt.show()
plt.plot(x,signals[0])
plt.show()
"""