# PyTorch implementation of NICE

Original paper:
  > [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)\
  > Laurent Dinh, David Krueger, Yoshua Bengio

This implementation replicates the experiment on the MNIST dataset.

A test-set log likelihood of 1933.89 was recorded after 70 epochs with the current hyperparameters. The original paper reported a similar test-set log likelihood, 1980.50.

To train this on your own system, install NumPy and PyTorch, edit config.py, and run train.py.

## Samples

![](/samples/samples1.png?raw=true)

![](/samples/samples2.png?raw=true)

![](/samples/samples3.png?raw=true)

![](/samples/samples4.png?raw=true)
