# Master

## my_utils.py
Needed for running the code on ML-servers as the GAN-code includes the package braindecoder.
Replace `from braindecode.datautil.iterators import get_balanced_batches` with `from my_utils import functions`

## /GAN

Code for
Hartmann, K. G., Schirrmeister, R. T., & Ball, T. (2018).
EEG-GAN: Generative adversarial networks for electroencephalograhic (EEG) brain signals.
Retrieved from https://arxiv.org/abs/1806.01875

## /VAE

Just testing varational autoencoder for image generation.
Currently configured for images using dataset from https://kaggle.com/chetankv/dogs-cats-images
Can be configured for LFP/EEG in the future.


