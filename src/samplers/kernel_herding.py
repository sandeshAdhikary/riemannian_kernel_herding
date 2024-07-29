# import numpy as np
import autograd.numpy as np
from pymanopt.manifolds import Sphere, Euclidean
from pymanopt.manifolds import SymmetricPositiveDefinite as PositiveDefinite
from pymanopt.manifolds import Oblique
from pymanopt.manifolds.euclidean import Symmetric
# from pymanopt.manifolds import Stiefel as Orthogonal
from src.custom_manifolds.orthogonal import Orthogonal
from pymanopt.manifolds.special_orthogonal_group import SpecialOrthogonalGroup as Rotation
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient, NelderMead, ParticleSwarm, TrustRegions, SteepestDescent
from src.kernels.kernels import Laplacian
import pymanopt

import torch

from tqdm import trange, tqdm

# Define data type for manifolds
vector_manifolds = [Euclidean, Sphere]
matrix_manifolds = [PositiveDefinite, Symmetric, Rotation, Oblique, Orthogonal]

def attr_fn(x, X, w, kernel_fn):
    '''
    :param x: test point
    :param X: Reference samples
    :param w: weights
    :param kernel_fn: kernel function
    :return:
    '''

    if len(x.shape) < 2:
        x_ = x.reshape(1, -1)
    else:
        x_ = x
    # print(x_)
    out = kernel_fn(x_, X)

    if torch.is_tensor(out):
        raise (NotImplementedError("Torch tensors not supported"))
        # if not torch.is_tensor(w):
        #     w = torch.from_numpy(w)
        # out = out.matmul(w)
    else:
        out = np.matmul(out, w)
    return out[0]

def repulse_fn(x, X, kernel_fn):
    '''
    :param x: test point
    :param X: Reference samples
    :param w: weights
    :param kernel_fn: kernel function
    :return:
    '''

    if len(x.shape) < 2:
        x_ = x.reshape(1, -1)
    else:
        x_ = x

    out = kernel_fn(x_, X)
    if torch.is_tensor(out):
        raise (NotImplementedError("Torch tensors not supported"))
        # out = torch.matmul(out, (torch.ones(out.shape[1]))/out.shape[1])
    else:
        out = np.matmul(out, (np.ones(out.shape[1]))/out.shape[1])

    return out[0]

def riemannian_kernel_herding(kernel_fn, w, X, n_herd,
                              manifold, true_manifold,
                              verbose=False, maxiter=100000,
                              opt_algo="ConjugateGradient",
                              no_repulse = False
                              ):
    ## manifold: The manifold used for optimization
    ## true_manifold: The manifold on which the data actually exists
    ## e.g. true_manifold might be PositiveDefinite matrices but we want to use a Euclidean optimizer

    if opt_algo == "NelderMead":
        solver = NelderMead(maxiter=maxiter)
    elif opt_algo == "TrustRegions":
        solver = TrustRegions()
    elif opt_algo == "ConjugateGradient":
        solver = ConjugateGradient(maxiter=maxiter)
    elif opt_algo == "SteepestDescent":
        solver = SteepestDescent()
    elif opt_algo == "ParticleSwarm":
        solver = ParticleSwarm(maxiter=maxiter)
    else:
        raise(NotImplementedError("Unknown Optimization algorithm"))

    verbosity = 2 if verbose else 0

    # Check if true data is vectors or matrices
    true_data_vector = True if type(true_manifold) in vector_manifolds else False
    true_data_matrix = True if type(true_manifold) in matrix_manifolds else False
    assert true_data_vector or true_data_matrix, "Data must either be vector or matrix"

    # Check if opt data is vectors or matrices
    opt_data_vector = True if type(manifold) in vector_manifolds else False
    opt_data_matrix = True if type(manifold) in matrix_manifolds else False
    assert opt_data_vector or opt_data_matrix, "Data must either be vector or matrix"

    if not type(manifold) == Euclidean:
        assert type(manifold) == type(true_manifold), "The optimization manifold must either be Euclidean or the same as the true manifold"

    # Our herded points should match the shape of the true data (vectors or matrices)
    if true_data_vector:
        X_herd = np.zeros((n_herd, X.shape[1]))
    else:
        X_herd = np.zeros((n_herd, X.shape[1], X.shape[2]))


    for idx in trange(n_herd, desc="Herding progress", leave=False):

        # If opt_data_matrix, then we'll be optimizing over manifold objects of shape [dim, dim]
        # But the kernel distance function will expect [1, dim, dim]
        if opt_data_matrix:
            # TODO: This is really messy, need a better way
            if idx == 0 or no_repulse:
                @pymanopt.function.Autograd
                def cost(x):
                    out = -attr_fn(x[np.newaxis, :], X, w, kernel_fn)
                    if np.isnan(out):
                        raise(ValueError("Cost is NaN!"))
                    return out
            else:
                @pymanopt.function.Autograd
                def cost(x):
                    out = -attr_fn(x[np.newaxis,:], X, w, kernel_fn) + repulse_fn(x[np.newaxis, :], X_herd[0:idx, :], kernel_fn)
                    if np.isnan(out):
                        raise(ValueError("Cost is NaN!"))
                    return out
        else:
            if idx == 0 or no_repulse:
                @pymanopt.function.Autograd
                def cost(x):
                    out = -attr_fn(x, X, w, kernel_fn)
                    if np.isnan(out):
                        raise(ValueError("Cost is NaN!"))
                    return out
            else:
                @pymanopt.function.Autograd
                def cost(x):
                    out = -attr_fn(x, X, w, kernel_fn) + repulse_fn(x, X_herd[0:idx, :], kernel_fn)
                    if np.isnan(out):
                        raise(ValueError("Cost is NaN!"))
                    return out

        X_herd[idx, :] = solver.solve(Problem(manifold=manifold, cost=cost, verbosity=verbosity))

    return X_herd

def run_herding(data,
                true_manifold,
                num_herd,
                kernel_type,
                optimization_type,
                bandwidth,
                data_weights=None,
                verbose=True,
                maxiter=10000,
                opt_algo="ConjugateGradient",
                no_repulse=False):
    '''
    :param data: 
    :param true_manifold: 
    :param num_herd: 
    :param kernel_type: ['Euclidean','Sphere','SPD']
    :param optimization_type: 
    :param bandwidth: 
    :param verose: 
    :param maxiter: 
    :return: 
    '''

    num_data = data.shape[0]
    dims = data.shape[-1]

    # Set up the kernel
    kernel = Laplacian(bandwidth=bandwidth, manifold=kernel_type)

    # Set up weights on data points. If not provided, use uniform weights
    w = data_weights if data_weights is not None else np.ones(num_data)/num_data

    # Set up the optimization manifold
    if optimization_type == "Euclidean":
        if type(true_manifold) in matrix_manifolds:
            # square matrices
            opt_manifold = Euclidean(1, dims, dims)
        else:
            # vectors
            opt_manifold = Euclidean(1, dims)
    else:
        opt_manifold = true_manifold


    # Herd new samples
    print("\t Herding...")
    X_herd = riemannian_kernel_herding(kernel.kernel_eval,
                                       w,
                                       data,
                                       num_herd,
                                       manifold=opt_manifold,
                                       true_manifold=true_manifold,
                                       verbose=verbose,
                                       maxiter=maxiter,
                                       opt_algo=opt_algo,
                                       no_repulse=no_repulse)

    return X_herd
