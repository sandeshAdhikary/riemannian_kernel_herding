import autograd.numpy as np
import torch

def isOblique(X):
    '''
    X.shape = (num_samples, dim, dim)
    :param X:
    :return:
    '''

    assert X.shape[1] == X.shape[2]

    if torch.is_tensor(X):
        raise(NotImplementedError("Torch tensors not supported"))
    else:
        err = np.max(1 - np.linalg.norm(X, axis=1))
        err = err < 1e-11 # True if sphere, Fase if not
        return err

def project(X):
    '''
    Project onto Oblioque manifold
    :param X: shape = (num_samples, dim, dim)
    :return:
    '''

    norms = np.linalg.norm(X, axis=1)
    return X/(norms[:, np.newaxis, :].repeat(X.shape[1], axis=1))

def geodesic(X,Y):
    '''
    Geodesic distance between X and Y
    X.shape = (num_x, dim, dim)
    Y.shape = (num_y, dim, dim)
    out = (num_x, num_y) matrix with distances
    :param X:
    :param Y:
    :return:
    '''

    # First, project the data onto the Oblique manifold if not already on it
    X_ = X if isOblique(X) else project(X)
    Y_ = Y if isOblique(Y) else project(Y)

    assert X_.dtype == np.double
    assert Y_.dtype == np.double

    assert isOblique(X_), "X is not on Oblique manifold, even after projection"
    assert isOblique(Y_), "Y is not on Oblique manifold, even after projection"

    clip_lim = 1 - 1e-10

    if torch.is_tensor(X_) and torch.is_tensor(Y_):
        raise (NotImplementedError("Torch tensors not supported"))
    else:
        dists = X_[:,np.newaxis,:] * Y_ # element-wise products across batches
        dists = np.sum(dists, axis=2)
        if len(dists[dists > 1]) != 0:
            dists[dists > 1] = 1

        assert np.max(dists) - 1 < 1e-10
        assert np.abs(np.min(dists)) - 1 < 1e-10
        # #
        dists = np.clip(dists, -clip_lim, clip_lim) # Clip to prevent NaNs in arc cos and gradient of arc cos

        dists = np.linalg.norm(np.arccos(dists), axis=2)
        # dists = np.einsum('ijk, lkn -> iljn', np.transpose(X_, (0,2,1)), Y_) # X^T Y
        # dists = dists.diagonal(axis1=-1, axis2=-2) # Get diagonal entries
        #
        # assert np.max(dists) - 1 < 1e-10
        # assert np.abs(np.min(dists)) - 1 < 1e-10
        # #
        # dists = np.clip(dists, -clip_lim, clip_lim) # Clip to prevent NaNs in arc cos and gradient of arc cos
        #
        # dists = np.arccos(dists)**2
        # dists = np.sum(dists, axis=2)
        #
        assert not np.any(np.isnan(dists)), "Distance is NaN"

        return dists
