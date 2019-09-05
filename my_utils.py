import numpy as np
import random

class my_utils():
    
    def get_balanced_batches(n_samples,batch_size):
        main_list = []
        samples = []
        for i in range(n_samples):
            samples.append(int(i))
        random.shuffle(samples)
        n_batches = int(n_samples/batch_size)
        for i in range(n_batches):
            tmp_list = []
            for j in range(batch_size):
                print(len(samples))
                tmp_list.append(samples[0])
                del samples[0]
                
            main_list.append(np.array(tmp_list))
        return main_list

