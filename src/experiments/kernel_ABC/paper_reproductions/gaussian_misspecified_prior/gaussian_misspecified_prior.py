import numpy as np
from numpy.random import multivariate_normal as y_cond_dist
from numpy.random import uniform as prior
from src.kr_abc.kr_abc import KernelRecursiveABC
from pymanopt.manifolds import Euclidean
import matplotlib.pyplot as plt
from src.manifold_utils.euclidean_utils import geodesic as euclidean_dist
from src.divergences.wasserstein import wasserstein_dist

np.random.seed(0)

## Load data
true_mean = np.load('data/true_theta.npy')
true_Sigma = np.load('data/true_Sigma.npy')
y_train = np.load('data/y_train.npy')
y_valid = np.load('data/y_valid.npy')

# Set up params
n_dims = true_mean.shape[0]
n_iters = 15
n_herd = 100
# prior_min = 1000
# prior_max = 2000
prior_min = 9e6
prior_max=9e7
true_manifold = Euclidean(n_dims) # The actual manifold for theta
opt_manifold_type = "Euclidean"
kernel_theta_type = "Euclidean"
kernel_y_type = "Euclidean"
y_dist_func = euclidean_dist
theta_dist_func = euclidean_dist

# Set up the prior sampler
def prior_sampler(num_samples):
    return prior(prior_min, prior_max, size=(num_samples, n_dims))

# Set up the prior sampler
def simulator(mu_list, Sigma=true_Sigma):
    '''
    Generate a sample from the target dist for each mu in mu_list
    :param mu_list:
    :return:
    '''

    y_sampled = np.zeros((mu_list.shape))
    for idx, mu in enumerate(mu_list):
        y_sampled[idx, :] = y_cond_dist(mu, Sigma, size=1)

    return y_sampled

# Set up hyperparams
reg = 1e-10
bandwidth_theta_scale = 0.173

# Run ABC
theta_estimates, theta_herds = KernelRecursiveABC(y_train, n_iters, prior_sampler, simulator,
                                                  None, kernel_y_type, kernel_theta_type,
                                                  opt_manifold_type, bandwidth_theta_scale,
                                                  n_herd, reg, true_manifold,
                                                  true_theta=true_mean.reshape(1, -1),
                                                  y_dist_func=y_dist_func,
                                                  theta_dist_func=theta_dist_func,
                                                  herding_algo="NelderMead")
# Save theta_estimates
np.save('results/theta_preds', theta_estimates)

## Get prediction errors from theta_estimates
y_pred_errs = np.zeros(theta_estimates.shape[0])
for pred_idx in np.arange(theta_estimates.shape[0]):

    # Generate new observations y_pred using MLE estimate of theta
    theta_mle = theta_estimates[pred_idx]
    y_pred = y_cond_dist(theta_mle, true_Sigma, size=y_valid.shape[0])

    y_pred_errs[pred_idx] = wasserstein_dist(y_valid, y_pred, dist_func=y_dist_func)


# Compute parameter error of theta_estimates wrt the true theta
param_errs = theta_estimates - np.expand_dims(true_mean,0).repeat(n_iters,0)
param_errs = np.linalg.norm(param_errs, axis=1)



fig, axs = plt.subplots(2, 2, figsize=(20, 10))

# Plot parameter errors
axs[0, 0].plot(param_errs, '-o')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xlabel("Iterations [0 to end]")
axs[0, 0].set_ylabel("Error")
axs[0, 0].set_title("Parameter Error")

axs[0, 1].plot(param_errs[1:], '-o')
axs[0, 1].set_yscale('log')
axs[0, 1].set_xlabel("Iterations [1 to end]")
axs[0, 1].set_ylabel("Error")
axs[0, 1].set_title("Parameter Error")


# Plot data errors
axs[1, 0].plot(y_pred_errs, '-o')
# axs[1, 0].set_yscale('log')
axs[1, 0].set_xlabel("Iterations [0 to end]")
axs[1, 0].set_ylabel("Error")
axs[1, 0].set_title("Prediction Error")

axs[1, 1].plot(y_pred_errs[1:], '-o')
# axs[1, 1].set_yscale('log')
axs[1, 1].set_xlabel("Iterations [1 to end]")
axs[1, 1].set_ylabel("Error")
axs[1, 1].set_title("Prediction Error")



# plt.savefig('results/gaussian_euclidean_{}dim/errs.png'.format(n_dim))
plt.savefig('results/errs.png')

