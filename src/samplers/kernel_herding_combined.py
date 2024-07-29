import pymanopt
import geoopt
import torch
from geoopt.optim import (RiemannianAdam)
from src.kernels.kernels import Laplacian
from src.divergences.wasserstein import wasserstein_over_samples
import matplotlib.pyplot as plt
from tqdm import trange
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient

torch.set_default_dtype(torch.double)


MANIFOLD_BACKENDS = ['pymanopt', 'geoopt']
NUMERIC_BACKENDS = ['numpy', 'pytorch']

class KernelHerder():

    def __init__(self, X, w, kernel, true_manifold, **kwargs):
        '''
        X: The data to herd around
        w: The weights applied to data points
        true_manifold: The true manifold on which samples live
        kernel: The kernel object
        Infer backend from the true_manifold provided
        If riemannian_opt is True, optimize over the true_manifold, else Euclidean
        '''

        self.X = X
        self.w = w
        self.kernel = kernel
        self.true_manifold = true_manifold
        self.manifold_backend = self.get_manifold_backend() # [geoopt, pymanopt]
        self.numeric_backend = self.get_numeric_backend(kwargs) # [numpy, pytorch]

        self.opt_init = kwargs.get('opt_init', 'manifold_random')

        if self.numeric_backend == "pytorch":
            self.X = torch.from_numpy(self.X) if not torch.is_tensor(self.X) else self.X
            self.w = torch.from_numpy(self.w) if not torch.is_tensor(self.w) else self.w

        if self.manifold_backend == "geoopt":
            self.herder = geoopt_herder(self.X, self.w, self.kernel, self.true_manifold,
                                        self.opt_init)
        else:
            self.herder = pymanopt_herder(self.X, self.w,
                                          self.kernel, self.true_manifold,
                                          self.numeric_backend)

    def run_herding(self, n_herd, riemannian_opt=True, **kwargs):
        '''
        Herd n_herd samples
        '''

        return self.herder.run_herding(n_herd, riemannian_opt, kwargs)

    def get_manifold_backend(self):
        '''
        Infer the backend of the manifold.
        '''

        # TODO: Change this. Not a reliable way. e.g. module name will be different for custom manifolds
        if hasattr(self.true_manifold, 'manifold_backend'):
            manifold_backend = self.true_manifold.manifold_backend
        # elif hasattr(self.manifold, 'n_manifolds'):
        #     manifold_backend = self.manifold.manifolds[0].__module__.split('.')[0]
        else:
            manifold_backend = self.true_manifold.__module__.split('.')[0]

        return manifold_backend

    def get_numeric_backend(self,kwargs):
        '''
        Set the numeric backend
        '''

        numeric_backend = kwargs['numeric_backend']

        if self.manifold_backend == "geoopt":
            assert numeric_backend == "pytorch", "Use 'pytorch' as numeric backend for geoopt"

        assert numeric_backend in NUMERIC_BACKENDS, \
            "Unrecognized numeric backend {}. Should be one of {}".format(numeric_backend, MANIFOLD_BACKENDS)

        return numeric_backend

class geoopt_herder():
    '''
    Herder to use when using the backend geoopt
    '''
    def __init__(self, X, w, kernel, true_manifold, opt_init="manifold_random"):
        self.X = X
        self.w = w
        self.opt_init = opt_init

        if self.w is None:
            # Assume uniform weights
            self.w = torch.ones(X.shape[0])/X.shape[0]

        assert torch.is_tensor(self.X), "The data should be a torch tensor"
        assert torch.is_tensor(self.w), "The weights should be a torch tensor"

        self.kernel = kernel
        self.true_manifold = true_manifold

    def attr_fn(self, x):
        '''
        x : sample to evaluate
        '''
        if len(x.shape) < 2:
            out = self.kernel.kernel_eval(x.reshape(1, -1), self.X)
        else:
            out = self.kernel.kernel_eval(x, self.X)
        return out.matmul(self.w)

    def repulse_fn(self, x, X_herd):
        '''
        x: sample to evaluate
        X_herd: current set of herded samples
        '''

        if len(x.shape) < 2:
            out = self.kernel.kernel_eval(x.reshape(1, -1), X_herd)
        else:
            out = self.kernel.kernel_eval(x, X_herd)

        out = torch.matmul(out, (torch.ones(out.shape[1])) / out.shape[1])

        return out[0]

    def run_herding(self, num_herd, riemannian_opt, param_dict):

        lr = param_dict.get("lr", 1e-2)
        num_epochs = param_dict.get("num_epochs", 1000)

        # Initialize X_herd with zeros
        X_herd = torch.repeat_interleave(torch.zeros_like(self.X[0, :]).unsqueeze(0),
                                         num_herd,
                                         dim=0)
        # Run herding for num_herd steps
        for idx in trange(num_herd, desc="Herding progress"):

            # Set up learnable params and optimizer
            if riemannian_opt:
                # Riemannian optimization
                if self.opt_init == "manifold_random":
                    x = self.true_manifold.random(self.X[0, :].shape).unsqueeze(0)
                elif self.opt_init == "data_random":
                    x = self.X[np.random.choice(np.arange(self.X.shape[0])), :].unsqueeze(0)
                else:
                    raise(ValueError("Unknown opt_init method"))
                x = geoopt.ManifoldParameter(x, manifold=self.true_manifold, requires_grad=True)
                optimizer = RiemannianAdam([x], lr=lr)
            else:
                # Euclidean optimization
                x = torch.rand(self.X[0, :].shape).unsqueeze(0)
                x = torch.nn.Parameter(x, requires_grad=True)
                optimizer = torch.optim.Adam([x], lr=lr)

            # Set up loss function
            if idx == 0:
                def loss_fn(x):
                    return -self.attr_fn(x)
            else:
                def loss_fn(x):
                    return -self.attr_fn(x) + self.repulse_fn(x, X_herd[0:idx, :])
            # Gradient descent for num_epochs
            # for t in trange(num_epochs, desc="\t Gradient descent progress", leave=False, position=1):
            for _ in range(int(num_epochs)):
                loss = loss_fn(x)
                # print(loss[0].detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print("Done")
            X_herd[idx, :] = x.detach()

        return X_herd

class pymanopt_herder():
    def __init__(self, X, w, kernel, true_manifold, numeric_backend):
        self.X = X
        self.w = w
        self.numeric_backend = numeric_backend

        if self.numeric_backend=="pytorch":
            self.nlib = torch
        elif self.numeric_backend == "numpy":
            self.nlib = np
        else:
            raise(ValueError("Unknown numeric_backend {}. Pick from {}".format(numeric_backend,
                                                                               NUMERIC_BACKENDS)))

        if self.w is None:
            # Assume uniform weights
            self.w = self.nlib.ones(X.shape[0]) / X.shape[0]

        self.kernel = kernel
        self.true_manifold = true_manifold

    def attr_fn(self, x):
        '''
        x : sample to evaluate
        '''
        if len(x.shape) < 2:
            out = self.kernel.kernel_eval(x.reshape(1, -1), self.X)
        else:
            out = self.kernel.kernel_eval(x, self.X)
        return self.nlib.matmul(out, self.w)

    def repulse_fn(self, x, X_herd):
        '''
        x: sample to evaluate
        X_herd: current set of herded samples
        '''

        if len(x.shape) < 2:
            out = self.kernel.kernel_eval(x.reshape(1, -1), X_herd)
        else:
            out = self.kernel.kernel_eval(x, X_herd)

        out = self.nlib.matmul(out, (self.nlib.ones(out.shape[1])) / out.shape[1])

        return out[0]

    def run_herding(self, num_herd, riemannian_opt, param_dict):

        opt_manifold = self.true_manifold if riemannian_opt else pymanopt.manifolds.Euclidean(1, *self.X.shape[1:])

        # Initiliaze the solver
        num_iters = param_dict.get("num_epochs", 1000)
        solver = ConjugateGradient(maxiter=num_iters)
        # solver = NelderMead(maxiter=25)

        # Initialize X_herd with zeros
        if self.numeric_backend == "pytorch":
            X_herd = torch.repeat_interleave(torch.zeros_like(self.X[0, :]).unsqueeze(0),
                                             num_herd,
                                             dim=0)
        elif self.numeric_backend == "numpy":
            X_herd = np.expand_dims(np.zeros_like(self.X[0, :]), 0).repeat(num_herd, 0)
        else:
            raise(ValueError("Unknown numeric_backend"))

        # Run herding for num_herd steps
        for idx in trange(num_herd, desc="Herding progress"):
            # Set up the loss function
            if idx == 0:
                if self.numeric_backend == "numpy":
                    @pymanopt.function.Autograd
                    def loss_fn(x):
                        return -self.attr_fn(x)[0]
                elif self.numeric_backend == "pytorch":
                    @pymanopt.function.PyTorch
                    def loss_fn(x):
                        return -self.attr_fn(x)[0]
            else:
                if self.numeric_backend == "numpy":
                    @pymanopt.function.Autograd
                    def loss_fn(x):
                        out =  -self.attr_fn(x) + self.repulse_fn(x, X_herd[0:idx, :])
                        return out[0]
                elif self.numeric_backend == "pytorch":
                    @pymanopt.function.PyTorch
                    def loss_fn(x):
                        out = -self.attr_fn(x) + self.repulse_fn(x, X_herd[0:idx, :])
                        return out[0]


            # Solve loss function and herd points
            herded_sample = solver.solve(Problem(manifold=opt_manifold,
                                                  cost=loss_fn,
                                                  verbosity=0))
            herded_sample = herded_sample if self.numeric_backend == "numpy" else torch.from_numpy(herded_sample)

            X_herd[idx, :] = herded_sample


        return X_herd

if __name__ == "__main__":

    np.random.seed(10)
    torch.manual_seed(10)

    dims = 3
    bandwidth = 10.0
    num_data = 150
    w = torch.rand(num_data)
    w[0:10] = 1.0 # There isn't much difference for uniformly distributed points
    w = w/torch.sum(w)

    # manifold_name = "Sphere"
    # ## Geoopt Manifold setup
    # true_manifold = geoopt.Sphere()
    # X = true_manifold.random((num_data, dims))
    ## Pymanopt setup: numpy
    # pymanopt_backend = "pytorch"
    # true_manifold = pymanopt.manifolds.sphere.Sphere(dims)
    # if pymanopt_backend == "numpy":
    #     X = np.stack([true_manifold.rand() for x in range(num_data)])
    # else:
    #     X = torch.stack([torch.from_numpy(true_manifold.rand()) for x in range(num_data)])

    # manifold_name = "SPD"
    # # Geoopt Manifold setup
    # pymanopt_backend = "pytorch"
    # # true_manifold = geoopt.SymmetricPositiveDefinite()
    # # X = true_manifold.random((num_data, dims, dims))
    # ## Pymanopt setup: numpy
    # numeric_backend = "pytorch"
    # true_manifold = pymanopt.manifolds.SymmetricPositiveDefinite(dims)
    # if pymanopt_backend == "numpy":
    #     X = np.stack([true_manifold.rand() for x in range(num_data)])
    # else:
    #     X = torch.stack([torch.from_numpy(true_manifold.rand()) for x in range(num_data)])

    # manifold_name = "Orthogonal"
    # # Geoopt Manifold setup
    # true_manifold = geoopt.Stiefel()
    # X = true_manifold.random((num_data, dims, dims))
    # numeric_backend = "pytorch"
    # ## Pymanopt setup: numpy
    # pymanopt_backend = "numpy"
    # numeric_backend = "numpy"
    # w = w.numpy()
    # from src.manifold_utils.ortho_utils import geodesic as ortho_dist
    # true_manifold = pymanopt.manifolds.Stiefel(dims, dims)
    # if pymanopt_backend == "numpy":
    #     X = np.stack([true_manifold.rand() for x in range(num_data)])
    # else:
    #     X = torch.stack([torch.from_numpy(true_manifold.rand()) for x in range(num_data)])

    manifold_name = "SO3"
    # # Geoopt Manifold setup
    # true_manifold = geoopt.Stiefel()
    # X = true_manifold.random((num_data, dims, dims))
    # ## Pymanopt setup: numpy
    pymanopt_backend = "numpy"
    numeric_backend = "numpy"
    w = w.numpy()
    from src.manifold_utils.so3_utils import geodesic as ortho_dist
    true_manifold = pymanopt.manifolds.SpecialOrthogonalGroup(dims)
    if pymanopt_backend == "numpy":
        X = np.stack([true_manifold.rand() for x in range(num_data)])
    else:
        X = torch.stack([torch.from_numpy(true_manifold.rand()) for x in range(num_data)])


    # Run herding
    num_herd = 75
    lr = 1e-3
    num_epochs = 25
    fig = plt.figure()
    for riemannian_opt in [True, False]:

        kernel_manifold = manifold_name if riemannian_opt else "Euclidean"
        kernel = Laplacian(bandwidth=bandwidth, manifold=kernel_manifold)

        kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                     numeric_backend=numeric_backend)

        X_herd = kernel_herder.run_herding(num_herd, riemannian_opt,
                                           lr=lr, num_epochs=num_epochs)

        wass = wasserstein_over_samples(X_herd, X,
                                 dist_func=ortho_dist,
                                 x_weights=None,
                                 y_weights=w, step=5, min_idx=5)

        plt.plot(wass, '-o', label="Riemannian Opt: {}".format(riemannian_opt))
    plt.legend()
    fig.savefig('test_errs.png')
    # plt.show()