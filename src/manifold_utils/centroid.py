import autograd.numpy as np
import torch
import geoopt
from geoopt.optim import (RiemannianAdam)
import pymanopt
from pymanopt.solvers import ConjugateGradient
from pymanopt.solvers import NelderMead
from pymanopt import Problem

MANIFOLD_BACKENDS = ['pymanopt', 'geoopt']
NUMERIC_BACKENDS = ['numpy', 'pytorch']

class RiemannianCentroid():
    def __init__(self, X, manifold, dist_func, w=None, **kwargs):
        self.X = X
        self.w = w
        self.manifold = manifold
        self.dist_func = dist_func

        self.manifold_backend = self.get_manifold_backend()  # [geoopt, pymanopt]
        self.numeric_backend = self.get_numeric_backend()  # [numpy, pytorch]

        if w is None:
            self.w = np.ones(X.shape[0])
            self.w = self.w/self.w.sum()

        if self.numeric_backend == "pytorch":
            self.X = torch.from_numpy(self.X) if not torch.is_tensor(self.X) else self.X
            self.w = torch.from_numpy(self.w) if not torch.is_tensor(self.w) else self.w

        if self.manifold_backend == "geoopt":
            self.centroid_optimizer = geoopt_centroid(self.X, self.w, self.manifold, dist_func)
        else:
            self.centroid_optimizer = pymanopt_centroid(self.X, self.w, self.manifold,dist_func,
                                                        self.numeric_backend)

    def get_centroid(self, **kwargs):
        '''
        Compute the weighted centroid of samples X with weights w
        :param X: samples to compute centroid of
        :param manifold: a manifold object (pymanopt or geoopt)
        :param w: weights placed on each sample. If none, assume uniform weights
        '''

        return self.centroid_optimizer.get_centroid(kwargs)


    def get_manifold_backend(self):
        '''
        Infer the backend of the manifold.
        '''
        # TODO: Change this. Not a reliable way. e.g. module name will be different for custom manifolds
        if hasattr(self.manifold, 'manifold_backend'):
            manifold_backend = self.manifold.manifold_backend
        # elif hasattr(self.manifold, 'n_manifolds'):
        #     manifold_backend = self.manifold.manifolds[0].__module__.split('.')[0]
        else:
            manifold_backend = self.manifold.__module__.split('.')[0]

        return manifold_backend

    def get_numeric_backend(self):
        '''
        Set the numeric backend
        '''

        numeric_backend = "pytorch" if torch.is_tensor(self.X) else "numpy"

        # numeric_backend = kwargs.get('numeric_backend','pytorch')
        #
        # if self.manifold_backend == "geoopt":
        #     assert numeric_backend == "pytorch", "Use 'pytorch' as numeric backend for geoopt"
        #
        # assert numeric_backend in NUMERIC_BACKENDS, \
        #     "Unrecognized numeric backend {}. Should be one of {}".format(numeric_backend,
        #                                                                   MANIFOLD_BACKENDS)

        return numeric_backend


class geoopt_centroid():
    def __init__(self, X, w, manifold, dist_func):
        self.X = X
        self.w = w
        self.dist_func = dist_func

        if self.w is None:
            # Assume uniform weights
            self.w = torch.ones(X.shape[0]) / X.shape[0]

        self.X = self.X if torch.is_tensor(self.X) else torch.from_numpy(self.X)
        self.w = self.w if torch.is_tensor(self.w) else torch.from_numpy(self.w)

        self.manifold = manifold

    def loss_fn(self, x):
        '''
        centroid = argmin_x \sum_i [ w_i (C(x, X_i)**2) ]
        '''
        return self.w.matmul(self.dist_func(self.X, x)**2)

    def get_centroid(self, opt_params):
        lr = opt_params.get('adam_lr', 1e-2)
        epochs = opt_params.get('adam_epochs', 25)
        opt_init = opt_params.get('opt_init', 'manifold_random')

        if opt_init == "manifold_random":
            x = self.manifold.random(self.X[0, :].shape).unsqueeze(0)
        elif opt_init == "data_random":
            x = self.X[np.random.choice(np.arange(self.X.shape[0])), :].unsqueeze(0)
        else:
            raise (ValueError("Unknown opt_init method"))

        x = geoopt.ManifoldParameter(x, manifold=self.manifold, requires_grad=True)
        optimizer = RiemannianAdam([x], lr=lr)
        for epoch in range(int(epochs)):
            loss = self.loss_fn(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return x

class pymanopt_centroid():
    def __init__(self, X, w, manifold, dist_func, numeric_backend):
        self.X = X
        self.w = w
        self.dist_func = dist_func
        self.numeric_backend = numeric_backend

        if self.w is None:
            # Assume uniform weights
            self.w = torch.ones(X.shape[0]) / X.shape[0]
            if self.numeric_backend == "numpy":
                self.w = self.w.detach().numpy()

        self.manifold = manifold

    @pymanopt.function.PyTorch
    def loss_fn_torch(self, x):
        '''
        centroid = argmin_x \sum_i [ w_i (C(x, X_i)**2) ]
        '''
        return self.w.matmul(self.dist_func(self.X, x)**2)

    @pymanopt.function.Autograd
    def loss_fn_autograd(self, x):
        '''
        centroid = argmin_x \sum_i [ w_i (C(x, X_i)**2) ]
        '''
        return self.w.matmul(self.dist_func(self.X, x)**2)

    # def loss_fn(self, x):
    #     if self.numeric_backend == "pytorch":
    #         return self.loss_fn_torch(x)
    #     else:
    #         return self.loss_fn_autograd(x)

    def get_centroid(self, opt_params):
        numiters = opt_params.get('adam_epochs', 1000)
        solver = ConjugateGradient(maxiter=numiters)
        if torch.is_tensor(self.X):
            @pymanopt.function.PyTorch
            def loss_fn(self, x):
                '''
                centroid = argmin_x \sum_i [ w_i (C(x, X_i)**2) ]
                '''
                out = self.w.matmul(self.dist_func(self.X, x) ** 2)
                out = out[0]
                return out
        else:
            @pymanopt.function.Autograd
            def loss_fn(x):
                '''
                centroid = argmin_x \sum_i [ w_i (C(x, X_i)**2) ]
                '''
                out = self.w @ (self.dist_func(self.X, x) ** 2)
                out = out[0]
                return out

        return solver.solve(Problem(manifold=self.manifold,
                                    cost=loss_fn,
                                    verbosity=0))
