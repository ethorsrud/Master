import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce

data_path = os.path.normpath(os.getcwd()+os.sep+os.pardir+os.sep+"Dataset"+os.sep+"mouse412804")

lfps = ["mouse412804_spontaneous_0_lfp.npy",
        "mouse412804_spontaneous_1_lfp.npy",
        "mouse412804_spontaneous_2_lfp.npy",
        "mouse412804_spontaneous_3_lfp.npy",
        "mouse412804_spontaneous_4_lfp.npy",
        "mouse412804_spontaneous_5_lfp.npy"]

times = ["mouse412804_spontaneous_0_t.npy",
        "mouse412804_spontaneous_1_t.npy",
        "mouse412804_spontaneous_2_t.npy",
        "mouse412804_spontaneous_3_t.npy",
        "mouse412804_spontaneous_4_t.npy",
        "mouse412804_spontaneous_5_t.npy"]

#Channels of interest over all sessions
all_channels_of_interest = []     

for i_file in range(len(lfps)):
        if i_file!=0:
                n_channels = lfp.shape[1]
        lfp = np.load(data_path+os.sep+lfps[i_file])
        if i_file!=0:
                #Making sure all sessions includes the same channels
                assert(n_channels==lfp.shape[1])

        time = np.load(data_path+os.sep+times[i_file])
        sample_freq = time.shape[0]/(time[-1]-time[0])
        print("Sample frequency = ",sample_freq)
        print("Total time",time[-1]-time[0])
        stds = np.std(lfp,axis=0)
        mean = np.mean(lfp,axis=0)
        """
        plt.plot(range(lfp.shape[1]),stds)
        plt.xlabel("Channel")
        plt.ylabel("Std")
        plt.show()
        """
        std_threshold = 1000

        ch_of_interest = np.nonzero(stds>std_threshold)
        all_channels_of_interest.append(ch_of_interest)
        #lfp_of_interest = lfp.T[ch_of_interest].T


channelmax = np.max(all_channels_of_interest[0])
channelmin = np.min(all_channels_of_interest[0])
for array in all_channels_of_interest:
        if np.max(array)>channelmax:
                channelmax = np.max(array)
        if np.min(array)<channelmin:
                channelmin = np.min(array)

#Gives all the common channels with std>threshold
equal_channels = reduce(np.intersect1d,(channel for channel in all_channels_of_interest))

cur_file = 0
for i_file in range(len(lfps)):
        lfp = np.load(data_path+os.sep+lfps[i_file])
        time = np.load(data_path+os.sep+times[i_file])
        lfp_of_interest = lfp.T[equal_channels].T
        #The mean of the Fourier transform over all channels
        
        
        freqs = np.fft.rfftfreq(lfp_of_interest.shape[0],d=1./sample_freq)
        fft = np.abs(np.fft.rfft(lfp_of_interest,axis=0).mean(axis=1))
        """
        plt.plot(freqs,fft)
        plt.title("Mean frequency spectrum\n over all channels")
        plt.xlabel("hz")
        plt.show()
        """
        #The results from the mean fourier transform over all channels
        #shows that most activity happens from 0-250hz, meaning we can
        #downsample our data to a minimum of 500hz to catch all the frequencies
        #by following Nyquist-Shannon sampling theorem

        #DOWNSCALE
        downsampled_lfp = []
        new_sampfreq = 500
        new_time = np.linspace(time[0],time[-1],new_sampfreq*(time[-1]-time[0]))
        for i in range(lfp_of_interest.shape[1]):
                downsampled_lfp.append(np.interp(new_time,time,lfp_of_interest[:,i]))
        downsampled_lfp = np.array(downsampled_lfp).T
        print("Original shape:",lfp_of_interest.shape,"Downscaled shape:",downsampled_lfp.shape)

        if cur_file == 0:
                all_sessions = downsampled_lfp
        else:
                all_sessions = np.concatenate((all_sessions,downsampled_lfp))
        print("Concatenated result:",all_sessions.shape)
        """
        plt.plot(time,lfp_of_interest[:,0])
        plt.plot(new_time,downsampled_lfp[:,0])
        plt.title("2500hz sample rate downsampled to 500hz")
        plt.xlabel("seconds")
        plt.ylabel("amp")
        plt.legend(["Original LFP","Downsampled LFP"])
        plt.show()
        """
        #lfp_scaled = (lfp-mean)/stds
        cur_file+=1

np.save("All_channels_%iHz"%(int(new_sampfreq)),all_sessions)
print("-"*50)
print("New dataset created with %.1f Hz sampling rate"%new_sampfreq)
print("Total time of new dataset: %.2f seconds (%.2f minutes)"%(all_sessions.shape[0]/new_sampfreq,all_sessions.shape[0]/(60*new_sampfreq)))
print("Number of channels included: %i"%all_sessions.shape[1])
dataset_info = os.stat("All_channels_%iHz.npy"%(int(new_sampfreq))) 
print("Dataset size: %.2f mb"%(dataset_info.st_size/10**6))
print("-"*50)
