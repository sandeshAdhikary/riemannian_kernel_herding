import numpy as np
from numpy.random import multivariate_normal as y_cond_dist
from numpy.random import uniform as prior
from src.kr_abc.kr_abc import KernelRecursiveABC
from pymanopt.manifolds import Euclidean
import matplotlib.pyplot as plt
from src.manifold_utils.euclidean_utils import geodesic as euclidean_dist
from src.divergences.wasserstein import wasserstein_dist

np.random.seed(1)

## Load data
true_mean = np.load('data/true_theta.npy')
true_Sigma = np.load('data/true_Sigma.npy')
y_train = np.load('data/y_train.npy')
y_valid = np.load('data/y_valid.npy')

# Set up params
n_dims = true_mean.shape[0]
n_iters = 30
n_herd = 10
# prior_min = 9e6
# prior_max = 10e7
prior_min =-20.
prior_max=20.
true_manifold = Euclidean(n_dims) # The actual manifold for theta
opt_manifold_type = "Euclidean"
kernel_theta_type = "Euclidean"
kernel_y_type = "Euclidean"
y_dist_func = euclidean_dist
theta_dist_func = euclidean_dist
max_hypers_to_test = 10 # Test at most this many hyerparameter configs

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

regs = np.geomspace(10e-4, 1, num=50)
bandwidth_theta_scales = np.geomspace(2e-4, 2e4, num=50)
all_hyperparams = np.array(np.meshgrid(regs, bandwidth_theta_scales)).T.reshape(-1, 2)
idxes = np.random.permutation(np.arange(all_hyperparams.shape[0]))[0:max_hypers_to_test]
random_hyperparams = all_hyperparams[idxes]

tuning_errs = np.zeros(random_hyperparams.shape[0])

for idx,(reg, bandwidth_scale) in enumerate(random_hyperparams):
    # Get MLE estimate of theta using y_train
    theta_estimates, theta_herds = KernelRecursiveABC(y_train, n_iters, prior_sampler, simulator,
                                                      None, kernel_y_type, kernel_theta_type,
                                                      opt_manifold_type, bandwidth_scale,
                                                      n_herd, reg, true_manifold,
                                                      true_theta=true_mean.reshape(1,-1),
                                                      y_dist_func=y_dist_func,
                                                      theta_dist_func=theta_dist_func)

    # Generate new observations y_pred using MLE estimate of theta
    theta_mle = theta_estimates[-1]
    y_pred = y_cond_dist(theta_mle, true_Sigma, size=y_valid.shape[0])

    err = wasserstein_dist(y_valid, y_pred, dist_func=y_dist_func)
    tuning_errs[idx] = err

print("-----------Done----------")
best_param = random_hyperparams[np.argmin(tuning_errs)]
print("Best reg: {}, best bandwidth: {}".format(best_param[0], best_param[1]))
regs = random_hyperparams[:, 0]
bands = random_hyperparams[:, 1]
fig = plt.figure()
scat = plt.scatter(regs, bands, c=tuning_errs, cmap="seismic")
plt.xlim(min(regs)*0.1, max(regs)*10)
plt.xscale('log')
plt.colorbar(scat)
plt.xlabel("Regularizer")
plt.ylabel("Bandwidth")
plt.title("Best Params: reg: {}, bandwidth: {}".format(best_param[0], best_param[1]))
plt.savefig('tuning/{}Kernel_{}Optimization.png'.format(kernel_theta_type,
                                                 opt_manifold_type))
print("Done")

# Error is the discrepency between y_pred and y_valid

# Compute error of theta_estimates wrt the true theta
# errs = theta_estimates - np.expand_dims(true_theta,0).repeat(n_iters,0)
# errs = np.linalg.norm(errs, axis=1)
#
# fig, axs = plt.subplots(1,2, figsize=(20,10))
# axs[0].plot(errs, '-o')
# axs[0].set_yscale('log')
# axs[0].set_xlabel("Iterations [0 to end]")
# axs[0].set_ylabel("Error")
#
# axs[1].plot(errs[1:], '-o')
# axs[1].set_yscale('log')
# axs[1].set_xlabel("Iterations [1 to end]")
# axs[1].set_ylabel("Error")
#
#
# # plt.savefig('results/gaussian_euclidean_{}dim/errs.png'.format(n_dim))
# plt.savefig('krabc_errs.png')

