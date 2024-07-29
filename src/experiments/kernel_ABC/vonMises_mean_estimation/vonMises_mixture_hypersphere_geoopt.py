import numpy as np
import torch
# from src.kr_abc.kr_abc import KernelRecursiveABC
from src.kr_abc.kr_abc_combined import KernelRecursiveABC
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from src.distributions.von_mises import VonMisesMixture3D
from src.manifold_utils.euclidean_utils import geodesic as euclid_dist_func
import geoopt
from src.kernels.kernels import RBF, Laplacian, ProductKernel
from src.manifold_utils.product import get_submanifold_tensors

torch.set_default_dtype(torch.double)

np.random.seed(0)
torch.manual_seed(0)

seeds = [0] #, 1, 2, 3, 4]

# Set up manifolds
sphere = geoopt.Sphere()
dims = 4
num_manifolds = 10 # i.e. number of components in GMM
prod_dims = num_manifolds*dims
theta_manifold = geoopt.manifolds.product.ProductManifold( *((sphere, dims) for _ in range(num_manifolds)))

def sub_tensor_fn(x,idx):
    return get_submanifold_tensors(x, idx, theta_manifold)

y_manifold_name = "Euclidean"
theta_manifold_names = ["Sphere", "Sphere"]

true_theta = theta_manifold.random((prod_dims))
true_loc_list = torch.stack([get_submanifold_tensors(true_theta, idx, theta_manifold) for idx in np.arange(num_manifolds)])
true_concs = torch.tensor([1.0]*num_manifolds)
true_weights = torch.tensor([1./num_manifolds]*num_manifolds)
dist = VonMisesMixture3D(locs=true_loc_list.unsqueeze(1),
                         concs=true_concs,
                         weights=true_weights)

# Set up hyperparameters
num_data = 500
num_iters = 100
theta_bandwidths = [1.0, 1.0]
theta_reg = 1e-10
y_bandwidth = 1.0
num_herd = 50

# Set up the kernels
kernel_theta_one = Laplacian(bandwidth=theta_bandwidths[0], manifold=theta_manifold_names[0])
kernel_theta_two = Laplacian(bandwidth=theta_bandwidths[1], manifold=theta_manifold_names[1])
kernel_theta = ProductKernel([kernel_theta_one, kernel_theta_two], sub_tensor_fn)
kernel_y = Laplacian(bandwidth=y_bandwidth, manifold="Euclidean")

# Set up krabc parameters
theta_dist_func = euclid_dist_func
y_dist_func = euclid_dist_func
numeric_backend = "pytorch"
theta_project_func = None


# Set up the simulator function
def observation_simulator(theta_list):
    '''
    Simulate observations of a given list of thetas
    '''
    y_sampled = torch.zeros((theta_list.shape[0], dims))
    for idx in range(theta_list.shape[0]):
        theta = theta_list[idx, :]
        locs = torch.stack([sub_tensor_fn(theta, idm) for idm in range(num_manifolds)])
        target_dist = VonMisesMixture3D(locs=locs, concs=true_concs, weights=true_weights)
        y_sampled[idx, :] = target_dist.sample(1).detach().reshape(-1)
    return y_sampled

# Set up the prior theta sampler
def prior_theta_sampler(num_samples, scaler=100):
    prior_thetas = scaler*theta_manifold.random((num_samples, 1, prod_dims))
    return prior_thetas

# Run experiment across seeds
for seed in seeds:
    print("Running seed: {}".format(seed))

    # Generate data
    y = dist.sample(num_data)

    # Set up KRABC
    krabc = KernelRecursiveABC(y, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=False,
                               adapt_theta_bandwidth=False,
                               riemannian_opt=True,
                               y_dist_func=y_dist_func,
                               true_theta=true_theta,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=None)

    theta_estimates, est_errs, sampling_errs = krabc.run_estimator(num_iters=num_iters)

    fig, axs = plt.subplots(2)
    axs[0].plot(est_errs, '-o')
    axs[0].set_ylabel("Estimation Error")

    axs[1].plot(sampling_errs, '-x')
    axs[1].set_ylabel("Sampling Error")
    # axs[1].set_ylim(0,2)
    fig.savefig("results/vonMises_mixture_errs.png")
