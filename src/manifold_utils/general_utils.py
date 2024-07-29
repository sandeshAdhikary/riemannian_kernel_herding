import autograd.numpy as np
import torch

def multiprod(A, B):
    """
    Code from pymanopt:

    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # Code from: pymanopt.tools.multi

    if not torch.is_tensor(A) and not torch.is_tensor(B):
        # First check if we have been given just one matrix
        if A.ndim == 2:
            return np.dot(A, B)

        # Approx 5x faster, only supported by numpy version >= 1.6:
        return np.einsum('ijk,ikl->ijl', A, B)
    elif torch.is_tensor(A) and torch.is_tensor(B):
        raise(NotImplementedError("Torch tensors not supported"))
        # # First check if we have been given just one matrix
        # if A.ndim == 2:
        #     return torch.dot(A, B)
        #
        # # Approx 5x faster, only supported by numpy version >= 1.6:
        # return torch.einsum('ijk,ikl->ijl', A, B)
    else:
        raise (ValueError('Both tensors must be of same type (np or torch tensors)'))
