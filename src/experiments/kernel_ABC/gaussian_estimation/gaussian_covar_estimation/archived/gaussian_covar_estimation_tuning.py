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
from src.divergences.wasserstein import wasserstein_dist
from src.hyper_param_tuning.hyperband import hyperband


torch.set_default_dtype(torch.double)

np.random.seed(0)
torch.manual_seed(0)

seeds = [0] #, 1, 2, 3, 4]

# Set up the target distribution and data
dims = 3
true_theta = torch.rand((dims, dims))
true_theta = (true_theta.T).matmul(true_theta)
true_theta = true_theta + torch.eye(dims)
# true_theta = torch.eye(dims)
true_loc = torch.zeros(dims).reshape(1, -1)
dist = MultivariateNormal(true_loc, true_theta)

# Set up manifolds
theta_manifold_name = "SPD"
# theta_manifold = pymanopt.manifolds.SymmetricPositiveDefinite(dims)
theta_manifold = geoopt.SymmetricPositiveDefinite()
y_manifold_name = "Euclidean"


# Set up hyperparameters
num_data = 250
# num_iters = 3
# theta_bandwidth = 1.0
# theta_reg = 1e-6
# y_bandwidth = None
num_herd = 50
riemannian_opt = True

# hyperparams = {
#     'theta_bandwidth': None,
#     'theta_reg': 1e-6,
#     'y_bandwidth': None
# }

num_hypers = 4
theta_bandwidths = [None, 1.0, 1e1, 1e2, 1e-1, 1e-2]
theta_regs = [1e-8, 1e-4, 1e-2, 1.0]
y_bandwidths = [None, 1.0, 1e1, 1e2, 1e-1, 1e-2]
lr = [1e-3, 1e-4]
all_hyperparams = np.array(np.meshgrid(theta_bandwidths, theta_regs, y_bandwidths,lr)).T.reshape(-1, num_hypers)


def get_random_hyperparameter_configuration(all_hyperparams=all_hyperparams):
    idxes = np.random.choice(range(all_hyperparams.shape[0]), 1)
    return all_hyperparams[idxes].reshape(-1)

def run_then_return_val_loss(num_iters, hyperparameters):

    theta_bandwidth = hyperparameters[0]
    theta_reg = hyperparameters[1]
    y_bandwidth = hyperparameters[2]
    lr = hyperparameters[3]

    # Set up the kernels
    kernel_theta = Laplacian(bandwidth=theta_bandwidth, manifold=theta_manifold_name)
    kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

    # Set up krabc parameters
    theta_dist_func = spd_dist_func
    y_dist_func = euclid_dist_func
    numeric_backend = "pytorch"
    theta_project_func = spd_project_func


    # Set up the simulator function
    def observation_simulator(theta_list, num_samples=1):
        '''
        Simulate observations of a given list of thetas
        '''
        y_sampled = torch.zeros((theta_list.shape[0], num_samples, dims))
        for idx in np.arange(theta_list.shape[0]):
            target_dist = MultivariateNormal(true_loc, theta_list[idx, :])
            y_sampled[idx, :] = target_dist.sample((num_samples,)).detach().reshape(num_samples, dims)
        if num_samples == 1:
            y_sampled = y_sampled.reshape(-1, dims)
        return y_sampled

    # Set up the prior theta sampler
    def prior_theta_sampler(num_samples, scaler=1000):
        samples = scaler*torch.rand((num_samples, dims, dims))
        samples = torch.einsum('bij,bjk->bik', samples, samples.permute(0, 2, 1))
        samples += (0.001*torch.eye(dims)).unsqueeze(0).repeat(samples.shape[0], 1, 1)
        return samples

    # Generate data
    y = dist.sample((num_data,)).detach().reshape(num_data, -1)
    y_val = dist.sample((num_data,)).detach().reshape(num_data, -1)

    # Set up KRABC
    try:
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

        # Get Wasserstein distance error of samples wrt to random validation set
        simulated_samples = observation_simulator(torch.stack([theta_estimates[-1, :]]), num_samples=100).reshape(-1, dims)
        sampling_err = wasserstein_dist(simulated_samples, y_val, dist_func=y_dist_func)

        # Need to return negative of sampling loss since we want to minimize, not maximize
        return sampling_err
    except:
        return np.finfo(np.float).max

# Run Hyperband
T, val_losses = hyperband(get_random_hyperparameter_configuration,
              run_then_return_val_loss,
              max_iter=81,
              eta=3,
              min_iter=1,
              verbose=True)

np.save('tuning/gaussian_covar_estimation_tuned_hyperparams.npy', np.array(T))
np.save('tuning/gaussian_covar_estimation_tuned_vallosses.npy', np.array(val_losses))

print("Done")
