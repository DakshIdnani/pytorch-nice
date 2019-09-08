# PyTorch implementation of NICE

Original paper:
  > [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)\
  > Laurent Dinh, David Krueger, Yoshua Bengio

This implementation replicates the experiment on the MNIST dataset.
A similar test set data likelihood as the paper was achieved in 70 epochs with the current hyperparameters.

To train it on your own system, install NumPy and PyTorch, edit config.py, and run train.py.
