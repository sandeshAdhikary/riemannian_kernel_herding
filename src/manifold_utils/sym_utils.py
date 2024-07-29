import torch
import autograd.numpy as np

def project(X):
    '''
    Project X to the symmetric part of X: 0.5*(X + X.T)
    :param X:
    :return:
    '''

    return 0.5* (X + np.transpose(X, axes=(0,2,1)))

def isSym(X):
    '''
    Check if all matrices in X (shape: (num_mats, dim, dim)) are symmetric
    :param X:
    :return:
    '''
    return np.max(np.abs(X - np.transpose(X, axes=(0, 2, 1)))) < 1e-16



def geodesic(X, Y):
    '''
    Code adapted from Sasha Lambert
    Note: this is the same as Euclidean distance
    :param X:
    :param Y:
    :return:
    '''

    dims = X.shape[1]

    # Check if we need to project
    X_ = X if isSym(X) else project(X)
    Y_ = Y if isSym(Y) else project(Y)

    # Vectorize
    X_ = X_.reshape(X_.shape[0], -1)
    Y_ = Y_.reshape(Y_.shape[0], -1)


    if torch.is_tensor(X_):
        # Euclidean manifold
        XX = X_.matmul(X_.t())
        XY = X_.matmul(Y_.t())
        YY = Y_.matmul(Y_.t())
        # Note: The formula below computes squared euclidean distance
        out = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
        out = torch.sqrt(out.double() + 1e-8)
        assert not torch.any(torch.isnan(out))
        return out
    else:
        XX = np.matmul(X_, X_.T)
        XY = np.matmul(X_, Y_.T)
        YY = np.matmul(Y_, Y_.T)
        # Note: The formula below computes squared euclidean distance
        out = -2 * XY + np.expand_dims(np.diag(XX), 1) + np.expand_dims(np.diag(YY), 0)

        # TODO: Adding some noise to prevent square root of 0. This prevents gradients from being 0
        # The noise also accounts of tiny negative vals
        out_sqrt = np.sqrt(out + 1e-8)

        assert not np.any(np.isnan(out_sqrt))
        return out_sqrt/np.sqrt(dims)