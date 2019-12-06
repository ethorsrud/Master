import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

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
amplitudes = np.load(os.path.normpath(kilosort_path+os.sep+"amplitudes.npy")).astype(np.double)
channel_map = np.load(os.path.normpath(kilosort_path+os.sep+"channel_map.npy")).astype(np.int32)

#Testing to extract first 100 spikes with template 0 
spikes = np.where(spike_templates==np.array([0]))[0][:100]
template = templates[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.array([np.linspace(0,81,82) for i in range(20)]).T
y = np.array([np.linspace(0,19,20) for i in range(82)])
z = template[:,:20]
ax.plot_surface(x,y,z,alpha=0.7,cmap="Blues")




from scipy.signal import butter, filtfilt

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

data2 = butter_highpass_filter(data[:20,:200000],cutoff=400,fs=30000,order=5)
spike = data2[:20,int(spike_times[int(spikes[0])])-41:(int(spike_times[int(spikes[0])])+41)]
#spike = butter_highpass_filter(spike,cutoff=400,fs=30000,order=5)

z2 = spike.T
ax.plot_surface(x,y,z2/(amplitudes[spikes[0]]*70),cmap="Oranges",alpha=0.7)
print(amplitudes[spikes[0]])
"""
for i in range(20):
    plt.plot(template[:,i])
"""
plt.savefig("/home/eirith/Spike_output/template.png",dpi=400)
plt.close()
#Select the first channel where the template is not zero
selected_channel = np.where(template!=0)[1][0]
i=0
"""
for spike in spikes:
    plt.plot(data[0,int(spike_times[int(spike)])-41:(int(spike_times[int(spike)])+41)]/amplitudes[int(spike)])
    print(i)
    i+=1
plt.savefig("/home/eirith/Spike_output/test.png",dpi=400)
plt.close()
"""