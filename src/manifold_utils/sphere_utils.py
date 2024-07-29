import torch
import autograd.numpy as np

def geodesic(X, Y, scaler=1.0):
    ## Assumed input shapes: [num_samples, num_dims]
    ## Output shape: [num_samples_X, num_samples_Y]

    # First, project the data onto the unit sphere. Without that geodesic distance doesn't make sense
    X_ = project(X.reshape(-1, X.shape[-1]))
    Y_ = project(Y.reshape(-1, Y.shape[-1]))

    if torch.is_tensor(X_):
        assert X_.dtype == torch.double
    else:
        assert X_.dtype == np.double

    if torch.is_tensor(Y_):
        assert Y_.dtype == torch.double
    else:
        assert Y_.dtype == np.double

    assert isSphere(X_), "X doesnt have unit norm, even after projection"
    assert isSphere(Y_), "Y doesnt have unit norm, even after projection"

    clip_lim = 1 - 1e-10
    # We clip distas between (-0.999..., 0.999...) since exact -1 and 1 causes grad(arccos) to be undefined
    if torch.is_tensor(X_) and torch.is_tensor(Y_):
        dist = X_.matmul(Y_.t())
        dist = torch.clip(dist, -clip_lim, clip_lim)
        dist = torch.arccos(dist)
        assert not torch.any(torch.isnan(dist)), "Distance is NaN"
    elif not torch.is_tensor(X_) and not torch.is_tensor(Y_):
        dist = np.matmul(X_, Y_.T)
        dist = np.clip(dist, -clip_lim, clip_lim)
        dist = np.arccos(dist)
        assert not np.any(np.isnan(dist)), "Distance is NaN"
    else:
        raise(Exception("One of X or Y is a torch tensor and the other is not!"))

    dist = dist/np.pi # Scale between -1 and 1

    return dist*scaler # Apply additional scaling if provided


def isSphere(X):
    if torch.is_tensor(X):
        err = torch.norm(1 - torch.norm(X, dim=1))
    else:
        err = np.linalg.norm(1 - np.linalg.norm(X, axis=1))

    err = err < 1e-11 # True if sphere, Fase if not
    return err

def project(X):
    '''
    Project points X onto the unit sphere
    Divide the points by their norm
    '''
    if torch.is_tensor(X):
        return X / (torch.linalg.norm(X, axis=1).reshape(-1,1))
    else:
        return X / (np.linalg.norm(X, axis=-1).reshape(-1, 1))