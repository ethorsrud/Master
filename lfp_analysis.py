import numpy as np
import os
import matplotlib.pyplot as plt

data_path = os.path.normpath(os.getcwd()+os.sep+os.pardir+os.sep+"Dataset"+os.sep+"mouse412804")

lfps = ["mouse412804_spontaneous_0_lfp.npy",
        "mouse412804_spontaneous_1_lfp.npy",
        "mouse412804_spontaneous_2_lfp.npy",
        "mouse412804_spontaneous_3_lfp.npy",
        "mouse412804_spontaneous_4_lfp.npy",
        "mouse412804_spontaneous_5_lfp.npy"]

time = ["mouse412804_spontaneous_0_t.npy",
        "mouse412804_spontaneous_1_t.npy",
        "mouse412804_spontaneous_2_t.npy",
        "mouse412804_spontaneous_3_t.npy",
        "mouse412804_spontaneous_4_t.npy",
        "mouse412804_spontaneous_5_t.npy"]

lfp = np.load(data_path+os.sep+lfps[0])
stds = np.std(lfp,axis=0)
mean = np.mean(lfp,axis=0)
lfp_scaled = (lfp-mean)/stds

plt.plot(range(lfp.shape[1]),stds)
plt.xlabel("Channel")
plt.ylabel("Std")
plt.show()