import numpy as np
import random

class functions():
    def get_balanced_batches(n_samples,batch_size):
        """
        Returns a list of length n_batches arrays with indexes of samples
        
        example:
        3 batches, batch_size = 3
        [array([1,2,3]),array([1,3,4]),array([5,2,1])]
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
        inputs: 
            split: Length of each sample
            n_batches: Number of wanted batches
            channels: The channels you want batches from
            data: Dataset with shape [n_samples,channels]

        returns:
            batches of small random samples from one long sample
            with shape: [n_batches,channels,data,1]
        """
        data_len = data.shape[0]
        max_int = data_len-split
        for i in range(n_batches):
            print('Yolo mcYolosen')
        return 0

