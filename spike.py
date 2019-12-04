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
print(data.shape)