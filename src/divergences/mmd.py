import numpy as np
import torch

def mmd(X, Y, kernel_fn):
    '''
    X: [m, ndim]
    Y: [n, ndim]
    Compute the MMD based on empirical samples X and Y
    drawn from potentially different distributions.
    MMD will be computed with respect to the given kernel_fn
    '''

    m, xdim = X.shape
    n, ydim = Y.shape

    assert xdim == ydim, 'X and Y samples must have the same dimension'

    if torch.is_tensor(X) or torch.is_tensor(Y):
        mmd = torch.mean(kernel_fn(X, X))
        mmd -= 2*torch.mean(kernel_fn(X, Y))
        mmd += torch.mean(kernel_fn(Y, Y))
        mmd = torch.sqrt(mmd)
    else:
        mmd = np.mean(kernel_fn(X, X))
        mmd -= 2*np.mean(kernel_fn(X, Y))
        mmd += np.mean(kernel_fn(Y, Y))
        mmd = np.sqrt(mmd)

    return mmd

def mmd_over_samples(X, Y, kernel_fn):
    '''
    X: generated samples
    Y: target samples
    Get MMD for all incremental point sets in X
    '''

    assert X.shape[1] == Y.shape[1], "Target and reference distributions must have same dimension"
    mmds = [mmd(X[0:idx], Y, kernel_fn) for idx in np.arange(1, X.shape[0])]
    return mmds