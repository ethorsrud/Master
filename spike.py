import numpy as np
import matplotlib.pyplot as plt
import os

kilosort_path = os.path.normpath(os.getcwd()+3*(os.sep+os.pardir)+os.sep+"shared"+os.sep+"users"+os.sep+"eirith"+os.sep+"kilosort2_results"+os.sep)
dat_path = os.path.normpath(kilosort_path+os.sep+os.pardir+os.sep+"continuous.dat")
n_channels_dat = 384
data_len = 112933688832
dtype = 'int16'
offset = 0
sample_rate = 30000
hp_filtered = False

data = np.memmap(dat_path, dtype, "r", offset, (n_channels_dat, data_len//n_channels_dat))
spike_times = np.load(os.path.normpath(kilosort_path+os.sep+"spike_times.npy")).astype(np.uint64) #[nSpikes,]
spike_templates = np.load(os.path.normpath(kilosort_path+os.sep+"spike_templates.npy")).astype(np.uint32) #[nSpikes,]
templates = np.load(os.path.normpath(kilosort_path+os.sep+"templates.npy")).astype(np.float32) #[nTemplates,nTimePoints,nTempChannels]
templates_ind = np.load(os.path.normpath(kilosort_path+os.sep+"templates_ind.npy")).astype(np.float64) #[nTemplates,nTempChannels]
amplitudes = np.load(os.path.normpath(kilosort_path+os.sep+"amplitudes.npy")).astype(np.float64)

#Testing to extract first 100 spikes with template 0 
spikes = np.where(spike_templates==np.array([0]))[0][:100]
template = templates[0]
#Select the first channel where the template is not zero
selected_channel = np.where(template!=0)[1][0]

#for spike in spikes:
for i in range(247):
    plt.plot(data[i,int(spike_times[int(spikes[0])]):(int(spike_times[int(spikes[0])])+82)]/amplitudes[int(spikes[0])]+2*i)
    print(i)
plt.savefig("/home/eirith/Spike_output/test.png",dpi=400)
plt.close()