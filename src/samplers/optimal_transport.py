import geoopt
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import autograd.numpy as np
from src.manifold_utils.euclidean_utils import geodesic as euclid_dist
from src.manifold_utils.sphere_utils import geodesic as sphere_dist
from src.manifold_utils.spd_utils import geodesic as spd_dist
from tqdm import trange
import ot
from src.manifold_utils.centroid import RiemannianCentroid
from src.divergences.wasserstein import wasserstein_over_samples
from src.samplers.kernel_herding_combined import KernelHerder
from src.kernels.kernels import Laplacian
from multiprocessing import Pool, cpu_count
import pymanopt

torch.set_default_dtype(torch.double)

MANIFOLD_BACKENDS = ['pymanopt', 'geoopt']
NUMERIC_BACKENDS = ['numpy', 'pytorch']

# class OptimalTransporter():
#
#     def __init__(self, Xs, Xt, wt, dist_func, reg, ws=None, centroid_fn=None, **kwargs):
#         '''
#         Xs: The source data that is to be transported around
#         Xt: The samples from the target distribution
#         wt: The weights for samples from the target distribution
#         ws: The weights for samples from the source distribution
#         dist_func: The distance function used for optimal transport
#         '''
#
#         self.Xs, self.Xt, self.wt = Xs, Xt, wt
#         self.dist_func = dist_func
#         self.reg = reg
#         self.ws = np.ones(Xs.shape[0]) if ws is None else ws # Weights on source samples
#
#         self.manifold = kwargs.get('manifold', None)
#         self.num_processes = kwargs.get('num_processes', 1)
#         self.num_processes = min(self.num_processes, cpu_count() -1) #Keep one process open to prevent cpu freeze
#
#         if centroid_fn is None:
#             self.adam_lr = kwargs.get('adam_lr', 1e-3)
#             self.adam_epochs = kwargs.get('adam_epochs', 25)
#             self.centroid_fn = lambda x,w: get_centroid(x, w, manifold=self.manifold,
#                                                         dist_func=self.dist_func,
#                                                         adam_lr=self.adam_lr,
#                                                         adam_epochs=self.adam_epochs)
#             if self.num_processes > 1: print("Multiprocessing only available with custom centroid functions")
#             self.num_processes = 1
#         else:
#             self.centroid_fn = centroid_fn
#
#     def resample(self):
#         '''
#         Generate samples that match the empirical distribution of the weighted target samples
#         {wt, Xt}
#         '''
#         # Get cost matrix
#         M = self.cost_matrix(self.Xs, self.Xt)
#         # Get OT matrix
#         ot_mat = self.ot_matrix(self.ws, self.wt, M)
#         # Compute centroids (transported points) for each source sample
#         if self.num_processes > 1:
#             args = [(self.Xt, ot_mat[idx, :]) for idx in np.arange(self.Xs.shape[0])]
#             with Pool(self.num_processes) as pool:
#                 X_resamples = pool.starmap(self.centroid_fn, args)
#             X_resamples = np.stack(X_resamples).squeeze()
#         else:
#             X_resamples = np.zeros_like(self.Xs)
#             for idx in trange(X_resamples.shape[0], desc="Optimal transport progress"):
#                 X_resamples[idx, :] = self.centroid_fn(self.Xt, ot_mat[idx, :])
#
#         return X_resamples
#
#     def cost_matrix(self, Xs, Xt, normalize=True):
#         '''
#         Compute the cost_matrix between X and Xt using the dist_func provided
#         '''
#         M = self.dist_func(Xs, Xt)
#         if normalize: M /= M.max()
#         return M
#
#     def ot_matrix(self, ws, wt, M):
#         '''
#         Return the OT matrix using
#         ws: Weights on the source samples
#         wt: Weights on the target samples
#         M: cost/distance matrix between the samples
#         '''
#         return ot.sinkhorn(ws, wt, M, reg=self.reg)
#
# def get_centroid(X, w, manifold, dist_func, adam_lr, adam_epochs):
#     assert manifold is not None, "Provide a manifold object, or a custom centroid function"
#     centroid_optimizer = RiemannianCentroid(X, manifold, dist_func, w)
#     out = centroid_optimizer.get_centroid(adam_lr=adam_lr, adam_epochs=adam_epochs)
#     if torch.is_tensor(out):
#         out = out.detach().numpy()
#     return out
#
# if __name__ == "__main__":
#     # Generate data
#     ns = 150
#     nt = 150
#     dims = 4
#     adam_lr = 1e-3
#     adam_epochs = 25
#
#     # manifold = geoopt.Sphere()
#     # Xs = manifold.random((ns, dims)).numpy()
#     # Xt = manifold.random((nt, dims)).numpy()
#     # def dist_func(x, y): return (sphere_dist(x, y))
#     # manifold_name = "Sphere"
#
#
#     # manifold = geoopt.SymmetricPositiveDefinite()
#     # Xs = manifold.random((ns, dims, dims)).numpy()
#     # Xt = manifold.random((nt, dims, dims)).numpy()
#     # def dist_func(x, y): return(spd_dist(x,y))
#     # manifold_name="SPD"
#
#     manifold_name = "Orthogonal"
#     numeric_backend = "pytorch"
#     ## Pymanopt setup: numpy
#     pymanopt_backend = "numpy"
#     numeric_backend = "numpy"
#     w = None
#     from src.manifold_utils.ortho_utils import geodesic as ortho_dist
#     manifold = pymanopt.manifolds.Stiefel(dims, dims)
#     if pymanopt_backend == "numpy":
#         Xs = np.stack([manifold.rand() for x in range(ns)])
#         Xt = np.stack([manifold.rand() for x in range(nt)])
#     else:
#         Xs = torch.stack([torch.from_numpy(manifold.rand()) for x in range(ns)])
#         Xt = torch.stack([torch.from_numpy(manifold.rand()) for x in range(nt)])
#
#
#     def dist_func(x, y):
#         return ortho_dist(x, y)
#
#
#     # Generate weights
#     ws = np.ones((ns,)) / ns # uniform distribution on samples
#     wt = np.ones((nt,)) / nt  # uniform distribution on samples
#
#
#
#     # Define optimal transporter
#     reg = 1e-3
#     def centroid_fn(x,w):
#         out = get_centroid(x, w, manifold=manifold, dist_func=dist_func, adam_lr=adam_lr, adam_epochs=adam_epochs)
#         return out
#     optimal_transporter = OptimalTransporter(Xs, Xt, wt, dist_func, reg,
#                                              manifold=manifold,
#                                              adam_lr=adam_lr,
#                                              adam_epochs=adam_epochs,
#                                              centroid_fn=centroid_fn,
#                                              num_processes=100)
#
#     Xr = optimal_transporter.resample()
#     dists_ot = wasserstein_over_samples(Xr, Xt, dist_func=dist_func, step=1, min_idx=1)
#     plt.plot(dists_ot,'-o', label='Optimal Transport')
#     plt.show()


class OptimalTransporter():

    def __init__(self, Xs, Xt, dist_func, reg, ws=None, wt=None, centroid_fn=None, **kwargs):
        '''
        Xs: The source data that is to be transported around
        Xt: The samples from the target distribution
        wt: The weights for samples from the target distribution
        ws: The weights for samples from the source distribution
        dist_func: The distance function used for optimal transport
        '''

        self.Xs, self.Xt = Xs, Xt
        self.dist_func = dist_func
        self.reg = reg
        self.ws = np.ones(Xs.shape[0]) if ws is None else ws # Weights on source samples
        self.ws = self.ws/self.ws.sum()
        self.wt = np.ones(Xt.shape[0]) if wt is None else wt  # Weights on source samples
        self.wt = self.wt / self.wt.sum()

        self.manifold = kwargs.get('manifold', None)
        self.num_processes = kwargs.get('num_processes', 1)
        self.num_processes = min(self.num_processes, cpu_count() -1) #Keep one process open to prevent cpu freeze

        self.opt_init = kwargs.get('opt_init', 'manifold_random')

        if centroid_fn is None:
            self.adam_lr = kwargs.get('adam_lr', 1e-3)
            self.adam_epochs = kwargs.get('adam_epochs', 25)
            self.centroid_fn = lambda x, w: get_centroid(x, w, manifold=self.manifold,
                                                         dist_func=self.dist_func,
                                                         adam_lr=self.adam_lr,
                                                         adam_epochs=self.adam_epochs,
                                                         opt_init=self.opt_init)
            if self.num_processes > 1: print("Multiprocessing only available with custom centroid functions")
            self.num_processes = 1
        else:
            self.centroid_fn = centroid_fn

    def resample(self):
        '''
        Generate samples that match the empirical distribution of the weighted target samples
        {wt, Xt}
        '''
        # Get cost matrix
        M = self.cost_matrix(self.Xs, self.Xt)
        # Get OT matrix
        ot_mat = self.ot_matrix(self.ws, self.wt, M)
        # Compute centroids (transported points) for each source sample
        if self.num_processes > 1:
            args = [(self.Xt, ot_mat[idx, :]) for idx in np.arange(self.Xs.shape[0])]
            with Pool(self.num_processes) as pool:
                X_resamples = pool.starmap(self.centroid_fn, args)
            X_resamples = np.stack(X_resamples).squeeze()
        else:
            X_resamples = np.zeros_like(self.Xs)
            for idx in trange(X_resamples.shape[0], desc="Optimal transport progress"):
                X_resamples[idx, :] = self.centroid_fn(self.Xt, ot_mat[idx, :])

        return X_resamples

    def cost_matrix(self, Xs, Xt, normalize=True):
        '''
        Compute the cost_matrix between X and Xt using the dist_func provided
        '''
        M = self.dist_func(Xs, Xt)
        M = np.clip(M, 1e-10, 1e10)
        #TODO: Does this normalization cause overflow?
        if normalize: M /= M.max()
        M = np.clip(M, 1e-10, 1e10)
        return M

    def ot_matrix(self, ws, wt, M):
        '''
        Return the OT matrix using
        ws: Weights on the source samples
        wt: Weights on the target samples
        M: cost/distance matrix between the samples
        '''
        return ot.sinkhorn(ws, wt, M, reg=self.reg)

def get_centroid(X, w, manifold, dist_func, adam_lr, adam_epochs, opt_init='manifold_random'):
    assert manifold is not None, "Provide a manifold object, or a custom centroid function"
    centroid_optimizer = RiemannianCentroid(X, manifold, dist_func, w)
    out = centroid_optimizer.get_centroid(adam_lr=adam_lr, adam_epochs=adam_epochs,
                                          opt_init=opt_init)
    if torch.is_tensor(out):
        out = out.detach().numpy()
    return out

if __name__ == "__main__":
    # Generate data
    ns = 150
    nt = 150
    dims = 4
    adam_lr = 1e-3
    adam_epochs = 25

    manifold = geoopt.Sphere()
    Xs = manifold.random((ns, dims)).numpy()
    Xt = manifold.random((nt, dims)).numpy()
    def dist_func(x, y): return (sphere_dist(x, y))
    manifold_name = "Sphere"

    numeric_backend = "pytorch"
    w = None

    from src.manifold_utils.sphere_utils import geodesic as dist_func

    Xs = manifold.random((ns, dims))
    Xt = manifold.random((nt, dims))

    # Generate weights
    ws = np.ones((ns,)) / ns # uniform distribution on samples
    wt = np.ones((nt,)) / nt  # uniform distribution on samples


    # Define optimal transporter
    reg = 1e-2
    def centroid_fn(x,w):
        out = get_centroid(x, w, manifold=manifold, dist_func=dist_func, adam_lr=adam_lr, adam_epochs=adam_epochs)
        return out

    optimal_transporter = OptimalTransporter(Xs, Xt, dist_func, reg, wt,
                                             manifold=manifold,
                                             adam_lr=adam_lr,
                                             adam_epochs=adam_epochs,
                                             centroid_fn=centroid_fn,
                                             num_processes=1)

    Xr = optimal_transporter.resample()
    Xr = torch.from_numpy(Xr)
    dists_ot = wasserstein_over_samples(Xr, Xt, dist_func=dist_func,
                                        step=1, min_idx=10)
    plt.plot(dists_ot,'-o', label='Optimal Transport')
    plt.show()