import numpy as np
import numpy.linalg as lin

from src.samplers.kernel_herding import riemannian_kernel_herding as herd
from tqdm import trange

def filter(Ytest, data, kx, ky, xinit, xtrans, manifold, true_manifold,
           eps=0.001, delta=0.001, t_tracked=None, n_herd=None):

    """Estimate a posterior distribution using the kernel monte carlo filter.

    :param: Ytest: The observation input.
    
            This is a T x m matrix, where T is the number of time steps and m is
            the dimension of the observation space.
                  

    :param: data: The state and observation training data.
    
            This should compose of two outer-most elements, the state data and
            the observation data. Both of these objects should have n rows: the
            number of samples. The number of columns in each object should be
            the dimension of the respective space.


    :param: kx: The kernel for the state space.
    
            This object expects a method `kernel_eval` that takes two objects
            over the kernel input space and returns a kernel matrix with the
            respective dimensions corresponding to the input objects.

            That is, if the dimension of the inputs are n1 x d, and n2 x d
            respectively, then the returned object is n1 x n2.


    :param: ky: The kernel for the observation space: the same type as `kx`.


    :param: xinit: The initial state distribution.
    
            This is a distribution function that takes the number of samples as
            a parameter and returns a prediction for the state.

            The prediction returned should be n x d where n is the number of
            samples passed in and d is the dimension of the state space.


    :param: xtrans: The transitional state distribution.
    
            This is a distribution function that takes prior state samples as
            parameters and predicts the next state, which has the same shape as
            the input.


    :param: manifold: The manifold object to sample from.


    :param: true_manifold: the expected manifold object to sample from


    :param: eps: regularization term #TODO for what?


    :param: delta: regularization term #TODO for what?


    :param: t_tracked: The number of previous states to keep track of.
    
            This defaults to the total number of time steps in the test data.


    :param: n_herd: The number of samples to use when herding.
    
            This defaults to the same number of samples in `data`


    :return: A tuple of the weights and the prior state data.
    """
    # unpack data, throw error if packed in incompatable format
    if len(data) != 2:
        message = "`data` should have outer-most dimension = 2, " \
                  f"but dimension is {len(data)}"
        raise ValueError(message)
    X, Y = data

    n, d = X.shape      # number of samples from data
    T = len(Ytest)      # number of input observations
    if n_herd is None:  # default value of n_herd = n
        n_herd = n
    if t_tracked is None:
        t_tracked = T   # default value of t_tracked = T

    # define the kernel matrices over the state and observation spaces
    Gx = kx.kernel_eval(X, X) # state space kernel
    Gy = ky.kernel_eval(Y, Y) # observation space kernel

    # the weights associated with the posterior distribution
    W = []
    Xprior = []

    for t in trange(T, desc="Filtering progress", leave=False):

        # on first iteration sample using initial distribution instead of the
        # transition model
        if t == 0:
            # 1. Prediction Step (initial):
            # - estimate initial state samples
            Xprior.append(xinit(n_herd)) # TODO: what is dimension here??
        else:
            # 3. Resampling Step:
            # - eliminate small-weight samples and replicate large-weight ones
            Xbar = herd(kx.kernel_eval, W[t-1], X, n_herd,
                        manifold, true_manifold)

            # 1. Prediction Step (transitional):
            # - estimate the next state samples via transition model
            Xprior.append(xtrans(Xbar))
            # only keep track of last t_tracked elements
            if t >= t_tracked:
                Xprior.pop(0)

        # 2. Correction Step
        # - estimate the kernel mean of the posterior using kernel bayes

        # compute the prior mean
        #   X in R^{n, d} : Xprior[t] in R^{n_herd, d}
        #   kx(X, Xprior[t]) in R^{n, n_herd}
        #   sum along i = 1..n_herd, where i are rows of Xprior[t]
        #   here Xprior[t] is actually Xprior[-1]
        mpi = np.sum(kx.kernel_eval(X, Xprior[-1]), axis=1) / n_herd

        # compute vector of kernel representations of observations
        kY = ky.kernel_eval(Y, Ytest[t])

        # calculate the weights for the posterior mean
        W.append(bayes(kY, mpi, Gx, Gy, eps, delta))
        W[t] = W[t] / np.sum(W[t]) # normalize

        # only keep track of last t_tracked elements
        if t >= t_tracked:
            W.pop(0)

    # stack the lists into np arrays and squeeze out any trivial dimensions
    return np.squeeze(np.stack(W)), np.squeeze(np.stack(Xprior))

def bayes(kY, mpi, Gx, Gy, eps=0.001, delta=0.001):
    """Calculate the weights for the posterior kernel mean.

    :param kY: the kernel vectors of and individual y and all Yn observations
    :param mpi: the empirical prior kernel means
    :param Gx: the kernel matrix for the state data
    :param Gy: the kernel matrix for the observation data
    :return: the weights that define the function from the posterior kernel
             mean.

    The weights are computed using the kernel bayes rule as shown in Kanagawa
    et. al.
    """
    n = len(kY)

    # compute the eigenvalue? matrix using the regularized state kernel matrix
    # the eigenvalues? should map to the kernel mean under this transformation
    # so solve using the kernel mean
    Lambda = np.diag(lin.solve(Gx + n*eps*np.eye(n), mpi))

    # closed form 
    M = Lambda @ Gy # observation kernel matrix transformed by Lambda
    return M @ lin.solve(M**2 + delta*np.eye(n), Lambda @ kY)
