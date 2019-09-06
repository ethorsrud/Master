# Master

## todo
- Make the save_model and load_model work correctly
- Add batch normalization to the network aiming for faster improving results
- Downscale network aiming for time-reduction along with maintaining reasonable results

## my_utils.py
Needed for running the code on ML-servers as the GAN-code includes the package braindecoder which the ML-servers does not have.
Replace 
```python
from braindecode.datautil.iterators import get_balanced_batches
```
with 
```python
from my_utils import functions 
```
and instead use 
```python
functions.get_balanced_batches(n_samples,batch_size)
```
My version uses only `n_samples` and `batch_size` as input and should do the same trick as the one from braindecode. 

## /GAN

Code for
Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018).
EEG-GAN: Generative adversarial networks for electroencephalograhic (EEG) brain signals.
Retrieved from https://arxiv.org/abs/1806.01875

## /VAE

Just testing varational autoencoder for image generation.
Currently configured for images using dataset from https://kaggle.com/chetankv/dogs-cats-images
Can be configured for LFP/EEG in the future. Use `VAE_new` as your VAE


