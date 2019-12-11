import numpy as np
import matplotlib.pyplot as plt
import os
import spikeinterface.extractors as se
import spikeinterface.toolkit as st

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

seconds_of_data = 10
small_n_channels = 4
small_data = data[:small_n_channels,:seconds_of_data*sample_rate]
print("Small data loaded")
small_data = small_data.astype(np.float32)
print("Data shape:",small_data.shape)
fft = np.abs(np.fft.rfft(small_data))
plt.plot(np.linspace(0,15001,fft.shape[0]),fft)
plt.savefig("FFT_small_data.png")
plt.close()

geom = np.zeros((small_n_channels,2))
geom[:,0] = range(small_n_channels)
recording = se.NumpyRecordingExtractor(timeseries=small_data,geom=geom,sampling_frequency=sample_rate)
small_data = st.preprocessing.bandpass_filter(small_data,freq_min=300,freq_max=6000)
print("Yeah that worked ")
"""
fft = np.abs(np.fft.rfft(small_data))
plt.plot(np.linspace(0,15001,fft.shape[1]),fft)
plt.savefig("FFT_small_data_BP.png")
plt.close()
"""