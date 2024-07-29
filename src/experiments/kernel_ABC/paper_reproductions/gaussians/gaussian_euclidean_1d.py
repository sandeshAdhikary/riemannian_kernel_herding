import numpy as np
from numpy.random import multivariate_normal as y_cond_dist
from numpy.random import uniform as prior
from src.kr_abc.kr_abc import KernelRecursiveABC
from pymanopt.manifolds import Euclidean
from src.kernels.kernels import RBF
import matplotlib.pyplot as plt

np.random.seed(0)

# Set up the prior distribution over mean vectors
def prior_sampler_func(num_samples, n_dim=1, low=0.0, high=1.0):
    return prior(low, high, size=(num_samples, n_dim))

# Set up the target distribution/simulator
n_dim = 1
mu = np.array([0.])
Sigma = 40.*np.eye(mu.shape[0])
true_theta = mu # The theta we're trying to learn

# Set up the prior sampler
prior_min = 200.
prior_max = 300.

def prior_sampler(num_samples): return prior_sampler_func(num_samples, n_dim=n_dim,
                                                          low=prior_min, high=prior_max)

def target_dist_sampler(num_samples, mu, Sigma):
    return y_cond_dist(mu, Sigma, size=num_samples)

def simulator(mu_list):
    '''
    Generate a sample from the target dist for each mu in mu_list
    :param mu_list:
    :return:
    '''

    y_sampled = np.zeros((mu_list.shape))
    for idx, mu in enumerate(mu_list):
        y_sampled[idx, :] = target_dist_sampler(1, Sigma=Sigma, mu=mu)

    return y_sampled

# Generate training observations
num_obs = 1000
y = target_dist_sampler(num_obs, mu, Sigma)


# Set up the manifolds
true_manifold = Euclidean(mu.shape[0]) # The actual manifold for theta
opt_manifold_type = "Euclidean"

# Set up the kernel for observations
# Note: If we're using the median trick inside the ABC func, kernel_y will get refined every iteration using samples
# bandwidth_y = 1. #Initial bandwidth
# kernel_y = RBF(bandwidth=bandwidth_y, manifold="Euclidean")

# Set up the kernel type for thetas
kernel_theta_type = "Euclidean"
kernel_y_type = "Euclidean"
bandwidth_theta = 10.0

# Set up ABC parameters
n_iters = 20
n_herd = 150
reg = 1e-8

# Run recursive kernel ABC
theta_estimates, theta_herds = KernelRecursiveABC(y, n_iters, prior_sampler, simulator, None,
                    kernel_y_type, kernel_theta_type, opt_manifold_type, bandwidth_theta,
                    n_herd, reg, true_manifold, true_theta=mu)

# Compute error of theta_estimates wrt the true theta
errs = theta_estimates - np.expand_dims(true_theta,0).repeat(n_iters,0)
errs = np.linalg.norm(errs, axis=1)

fig, axs = plt.subplots(1,1)
axs.plot(errs, '-o')
axs.set_yscale('log')
axs.set_xlabel("Iterations")
axs.set_ylabel("Error")
plt.savefig('results/gaussian_euclidean_1d/errs.png')


fig, axs = plt.subplots(1,4, figsize=(20,10))
lim = prior_max + np.int(prior_max*0.1)
axs[0].hist(theta_herds[0, :], bins=np.arange(-lim, lim))
axs[0].set_title("0-th Iteration")
axs[1].hist(theta_herds[1, :], bins=np.arange(-lim, lim))
axs[1].set_title("1st Iteration")
axs[2].hist(theta_herds[np.int(theta_herds.shape[0]/2), :], bins=np.arange(-lim, lim))
axs[2].set_title("{}-th Iteration".format(np.int(theta_herds.shape[0]/2)))
axs[3].hist(theta_herds[np.int(theta_herds.shape[0]-1), :], bins=np.arange(-lim, lim))
axs[3].set_title("{}-th Iteration".format(np.int(theta_herds.shape[0])))
for ax in axs:
    ax.set_xlabel("Estimated Param Sample")
    ax.set_ylabel("Num Samples")
plt.savefig('results/gaussian_euclidean_1d/hist.png')

