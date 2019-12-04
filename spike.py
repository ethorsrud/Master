import numpy as np
import matplotlib.pyplot as plt
import os

kilosort_path = os.path.normpath(os.getcwd()+3*(os.sep+os.pardir)+os.sep+"shared"+os.sep+"users"+os.sep+"eirith"+os.sep+"kilosort2_results")
continuous_path = os.path.normpath(kilosort_path+os.sep+os.pardir)

print(kilosort_path)
print(continuous_path)

dat_path = 'continuous.dat'
n_channels_dat = 384
data_len = 112933688832
dtype = 'int16'
offset = 0
sample_rate = 30000
hp_filtered = False