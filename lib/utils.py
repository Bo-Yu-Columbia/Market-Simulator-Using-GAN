import pickle

import numpy as np
import torch


def sample_indices(dataset_size, batch_size):
    # This function returns a tensor of randomly sampled indices from a dataset of a given size. 
    # The number of indices is determined by the batch size. The indices are generated using numpy.random.choice 
    # function and then converted to a PyTorch tensor on the GPU using torch.from_numpy.
    
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()
