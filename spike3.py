import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KILOSORT2_PATH"] = os.path.normpath(os.getcwd()+os.sep+os.pardir+os.sep+"Kilosort2")
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
from scipy.signal import butter,lfilter
import spikeinterface.sorters as ss

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
amplitudes = np.load(os.path.normpath(kilosort_path+os.sep+"amplitudes.npy")).astype(np.double)#[nSpikes, ]
channel_map = np.load(os.path.normpath(kilosort_path+os.sep+"channel_map.npy")).astype(np.int32)#[n_channels]
channel_positions = np.load(os.path.normpath(kilosort_path+os.sep+"channel_positions.npy")).astype(np.float64)#[n_channels,2]
whitening_mat = np.load(os.path.normpath(kilosort_path+os.sep+"whitening_mat.npy")).astype(np.float64) #[n_channels,n_channels]


channel_map = channel_map[:15]
channel_positions = channel_positions[:15,:]
n_samples = 768
input_length = 8192
spike_data_small = data[channel_map,:input_length*n_samples].astype(np.int16)
print(spike_times[0:50])
print(templates_ind[0,:])

#recording = se.NumpyRecordingExtractor(timeseries=spike_data_small,geom=channel_positions,sampling_frequency=sample_rate)

#recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
#recording = st.preprocessing.notch_filter(recording, freq=1000, q=10)
#recording = st.preprocessing.resample(st.preprocessing.rectify(recording), 1000)
#recording = st.preprocessing.common_reference(recording, reference='median')

#sorting_KS2 = ss.run_kilosort2(recording, output_folder='tmp_KS2')

"""
selected_template = 0
n_samples = 768
input_length = 8192

spike_data_small = data[:15,:input_length*n_samples].T
whitening_mat_small = whitening_mat[:15,:15]

print(spike_data_small.shape)
temp_index = np.where(spike_templates==selected_template)[0]
spike_times = spike_times[temp_index]
spike_times = spike_times[spike_times<(input_length*n_samples)]
amplitudes = amplitudes[temp_index]
amplitudes = amplitudes[:spike_times.shape[0]]
#FILTERING
#b,a = butter(4,6000/(0.5*sample_rate),btype="low")
#spike_data_small = lfilter(b,a,spike_data_small,axis=0)
b,a = butter(6,150/(0.5*sample_rate),btype="high")
spike_data_small = lfilter(b,a,spike_data_small,axis=0)


spike_data_small = (whitening_mat_small@spike_data_small.T).T

for i in range(3):
    plt.plot(spike_data_small[(int(spike_times[i])-41):(int(spike_times[i])+41),1],linewidth=0.3,alpha=0.5,label="%i"%i)
    plt.plot(templates[0,:,0]*amplitudes[i],linewidth=0.5,alpha=0.5)
    print("Amp",amplitudes[i])
plt.legend()
plt.savefig("Template0_realdata_spikes.png",dpi=500)
plt.close()
"""
