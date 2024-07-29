import numpy as np
import torch
# from src.kr_abc.kr_abc import KernelRecursiveABC
from src.kr_abc.kr_abc_combined import KernelRecursiveABC
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from src.manifold_utils.spd_utils import geodesic as spd_dist_func
from src.manifold_utils.euclidean_utils import geodesic as euclid_dist_func
from src.manifold_utils.spd_utils import project as spd_project_func
import geoopt
from src.kernels.kernels import RBF, Laplacian
import pymanopt
torch.set_default_dtype(torch.double)

np.random.seed(0)
torch.manual_seed(0)

seeds = [0] #, 1, 2, 3, 4]

# Set up the target distribution and data
dims = 10
# true_loc = torch.zeros(dims).reshape(1, -1)
true_theta = torch.rand(dims).reshape(1, -1)
true_covar = 1*torch.eye(dims)
dist = MultivariateNormal(true_theta, true_covar)

# Set up manifolds
theta_manifold_name = "Euclidean"
# theta_manifold = pymanopt.manifolds.Euclidean(dims)
theta_manifold = geoopt.manifolds.Euclidean()
y_manifold_name = "Euclidean"


# Set up hyperparameters
num_data = 500
num_iters = 50
theta_bandwidth = None
theta_reg = 1e-10
y_bandwidth = None
num_herd = 50

# Set up the kernels
kernel_theta = Laplacian(bandwidth=theta_bandwidth, manifold=theta_manifold_name)
kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

# Set up krabc parameters
theta_dist_func = euclid_dist_func
y_dist_func = euclid_dist_func
numeric_backend = "pytorch"
# theta_project_func = None


# Set up the simulator function
def observation_simulator(theta_list):
    '''
    Simulate observations of a given list of thetas
    '''
    y_sampled = torch.zeros((theta_list.shape[0], true_theta.shape[1]))
    for idx in np.arange(theta_list.shape[0]):
        target_dist = MultivariateNormal(theta_list[idx, :], true_covar)
        y_sampled[idx, :] = target_dist.sample((1,)).detach().reshape(-1)
    return y_sampled

# Set up the prior theta sampler
def prior_theta_sampler(num_samples):
    prior_loc = torch.zeros(dims).reshape(1, -1)
    prior_covar = torch.eye(dims)
    prior_dist = MultivariateNormal(prior_loc, prior_covar)
    return prior_dist.sample((num_samples,)).detach().reshape(-1,dims)


# Run experiment across seeds
for seed in seeds:
    print("Running seed: {}".format(seed))

    # Generate data
    y = dist.sample((num_data,)).detach().reshape(num_data, -1)

    # Set up KRABC
    krabc = KernelRecursiveABC(y, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=True,
                               adapt_theta_bandwidth=True,
                               riemannian_opt=True,
                               y_dist_func=y_dist_func,
                               true_theta=true_theta,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=None)

    theta_estimates, est_errs = krabc.run_estimator(num_iters=num_iters)

    fig = plt.figure()
    plt.plot(est_errs, '-o')
    plt.xlabel("Iterations")
    plt.ylabel("Estimation Error")
    fig.savefig("results/gaussian_mean_estimation.png")
