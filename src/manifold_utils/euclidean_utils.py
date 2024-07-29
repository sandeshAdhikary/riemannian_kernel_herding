import torch
import autograd.numpy as np

def geodesic(X, Y):
    '''
    Code adapted from Sasha Lambert
    :param X:
    :param Y:
    :return:
    '''
    if len(X.shape) == 2:
        # The data consists of vectors
        X_ = X
    else:
        # Convert the data into vectors first
        X_ = X.reshape(X.shape[0], -1)

    if len(Y.shape) == 2:
        Y_ = Y
    else:
        # Vectorize
        Y_ = Y.reshape(Y.shape[0], -1)

    if torch.is_tensor(X_) and torch.is_tensor(Y_):
        # # Euclidean manifold
        XX = X_.matmul(X_.t())
        XY = X_.matmul(Y_.t())
        YY = Y_.matmul(Y_.t())
        # # Note: The formula below computes squared euclidean distance
        out = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
        # Adding some noise to prevent square root of 0. This prevents gradients from being 0
        # The noise also accounts of tiny negative vals
        out = torch.sqrt(out.double() + 1e-8)
        assert not torch.any(torch.isnan(out))
    elif not torch.is_tensor(X_) and not torch.is_tensor(Y_):
        XX = np.matmul(X_, X_.T)
        XY = np.matmul(X_, Y_.T)
        YY = np.matmul(Y_, Y_.T)
        # Note: The formula below computes squared euclidean distance
        out = -2 * XY + np.expand_dims(np.diag(XX), 1) + np.expand_dims(np.diag(YY), 0)

        # Adding some noise to prevent square root of 0. This prevents gradients from being 0
        # The noise also accounts of tiny negative vals
        out = np.sqrt(out + 1e-8)
        assert not np.any(np.isnan(out))
    else:
        raise(ValueError("Input arrays X has type {} and Y as type {}".format(type(X), type(Y))))

    return out/np.sqrt(X.shape[1])
