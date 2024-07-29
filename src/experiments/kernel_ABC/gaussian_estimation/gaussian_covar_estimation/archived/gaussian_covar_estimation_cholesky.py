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
from src.divergences.wasserstein import wasserstein_over_samples

torch.set_default_dtype(torch.double)

np.random.seed(0)
torch.manual_seed(0)

seeds = [1, 2, 3, 4, 5, 6, 7, 8] #, 1, 2, 3, 4]
# seeds = [0,1]

def vec_to_chol(vec, dims):
    # mat_dim = int(vec.shape[0]/2)
    xc = torch.cat([vec[dims:], vec.flip(dims=[0])])
    y = xc.view(dims, dims)
    return torch.tril(y)

def chol_to_spd(chol):
    return chol.matmul(chol.T)

def num_lower_triu_elems(n):
    return int((n * (n - 1)) / 2) + n


# Set up the target distribution and data
dims = 3
chol_vec_dim = num_lower_triu_elems(dims)

true_theta = torch.rand(chol_vec_dim) # Vector with lower triu elems
try:
    torch.cholesky(chol_to_spd(true_theta))
except:
    raise(ValueError("True theta is not SPD"))
# true_theta = torch.rand((dims, dims))
# true_theta = (true_theta.T).matmul(true_theta)
# true_theta = true_theta + torch.eye(dims)
# true_theta = torch.cholesky(true_theta)
# true_theta = true_theta[true_theta > 0]
# true_loc = torch.zeros(dims).reshape(1, -1)
dist = MultivariateNormal(true_loc, chol_to_spd(vec_to_chol(true_theta, dims)))

# Set up manifolds
theta_manifold_name = "Euclidean"
# theta_manifold = pymanopt.manifolds.Euclidean(chol_vec_dim)
theta_manifold = geoopt.Euclidean()
y_manifold_name = "Euclidean"

# Set up hyperparameters
num_data = 200
num_iters = 10
theta_bandwidth = 10.0
theta_reg = 1e-8
y_bandwidth = 0.1
num_herd = 50
riemannian_opt = False
lr = 0.0001

# Set up the kernels
kernel_theta = Laplacian(bandwidth=theta_bandwidth, manifold=theta_manifold_name)
kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

# torch.tril_indices(10,10)

# Set up krabc parameters
theta_dist_func = euclid_dist_func
y_dist_func = euclid_dist_func
numeric_backend = "pytorch"
theta_project_func = None


# Set up the simulator function
def observation_simulator(theta_list, num_samples=1):
    '''
    Simulate observations of a given list of thetas
    '''
    y_sampled = torch.zeros((theta_list.shape[0], num_samples, dims))
    for idx in np.arange(theta_list.shape[0]):
        covar = chol_to_spd(vec_to_chol(theta_list[idx, :], dims))
        target_dist = MultivariateNormal(true_loc, covar)
        y_sampled[idx, :] = target_dist.sample((num_samples,)).detach().reshape(num_samples, dims)
    if num_samples == 1:
        y_sampled = y_sampled.reshape(-1, dims)
    return y_sampled

# Set up the prior theta sampler
def prior_theta_sampler(num_samples, scaler=1000):
    samples = scaler*torch.rand((num_samples, chol_vec_dim))
    return samples


# Run experiment across seeds
sampling_errs = []
for seed in seeds:
    print("Running seed: {}".format(seed))

    # Generate data
    y = dist.sample((num_data,)).detach().reshape(num_data, -1)
    y_test = dist.sample((num_data,)).detach().reshape(num_data, -1)


    # Set up KRABC
    krabc = KernelRecursiveABC(y, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=True,
                               adapt_theta_bandwidth=True,
                               riemannian_opt=riemannian_opt,
                               y_dist_func=y_dist_func,
                               true_theta=true_theta,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=theta_project_func,
                               lr=lr)

    theta_estimates, est_errs, _ = krabc.run_estimator(num_iters=num_iters)
    # Get Wasserstein distance error of samples wrt validation set
    simulated_samples = observation_simulator(torch.stack([theta_estimates[-1, :]]), num_samples=num_data).reshape(-1, dims)
    sampling_err = wasserstein_over_samples(simulated_samples, y_test, dist_func=y_dist_func, step=5, min_idx=3)
    sampling_errs.append(sampling_err)
    print("Final sampling err: {}".format(sampling_err[-1]))
    # fig, axs = plt.subplots(2)
    # axs[0].plot(est_errs, '-o')
    # axs[0].set_ylabel("Estimation Error")
    #
    # axs[1].plot(sampling_errs, '-x')
    # axs[1].set_ylabel("Sampling Error")
    # # axs[1].set_ylim(0,2)
    # fig.savefig("results/errs_cholesky.png")

np.save('results/gaussian_covar_estimation_cholesky_results.npy', sampling_errs)