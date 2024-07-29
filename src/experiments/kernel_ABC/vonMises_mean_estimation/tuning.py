import numpy as np
import torch
from src.distributions.von_mises import VonMises3D
from pymanopt.manifolds import Euclidean, Sphere
from src.kr_abc.kr_abc import KernelRecursiveABC
import matplotlib.pyplot as plt
from src.manifold_utils.sphere_utils import geodesic as sphere_dist
from src.manifold_utils.sphere_utils import project as sphere_project


seed = 2000


# Set up the target distribution and data
dims = 6
loc = np.zeros(dims).reshape(1,-1)
loc[:, 0] = 1.
loc = loc / np.linalg.norm(loc)  # Normalized so its on the sphere
true_loc = loc
conc = np.array([5.0]).reshape(1, -1)  # Defines concentration around loc
dist = VonMises3D(loc, conc)
y_dist_func = sphere_dist
y_project_func = sphere_project

def target_simulator(mu_list):
    '''
    Generate samples from the simulator for a list of estimated means mu_list
    :param mu_list: (num, dims) an array with num estimated means of dimension dim
    :return:
    '''

    y_sampled = np.zeros((mu_list.shape))
    for idx in np.arange(mu_list.shape[0]):
        mu = sphere_project(mu_list[idx,:].reshape(1,-1))
        target_dist = VonMises3D(mu, conc) # Use the same conc as target
        y_sampled[idx, :] = target_dist.sample(1).reshape(-1).double().numpy()

    return y_sampled


# Generate data
num_data = 100
np.random.seed(seed)
torch.manual_seed(seed)
y = dist.sample(num_data).numpy().reshape(-1, dims)


# Set up prior sampler
prior_loc = np.ones(dims).reshape(1, -1)
prior_loc = prior_loc / np.linalg.norm(prior_loc)
prior_conc = conc # Same conc as target distribution
prior_dist = VonMises3D(prior_loc, conc)

def prior_sampler(num_samples):
    return prior_dist.sample(num_samples).reshape(num_samples, -1).double().numpy()

# Set up the true manifold
true_manifold = Sphere(dims)

exps = [["Euclidean", "Euclidean"],
        ["Euclidean", "Sphere"],
        ["Sphere", "Euclidean"],
        ["Sphere", "Sphere"]]

# exps = [exps[3]]


# Set up hyperparameters to search over
regs = np.array([1e-10, 1e-8, 1e-4, 1e-2])
# bandwidths = np.geomspace(0.001, 1000, 30)
bandwidths = np.array([1e-4,1e-3,1e-2,1e-1,1e1,1e2])
all_hyperparams = np.array(np.meshgrid(regs, bandwidths)).T.reshape(-1, 2)
num_hypers = 20  # Only try this many hyperparams
idxes = np.random.permutation(np.arange(all_hyperparams.shape[0]))[0:num_hypers]
random_hyperparams = all_hyperparams[idxes]

# Start tuning
n_iters = 50
n_herd = 100  # Should be the same as n_herd used in experiment
for exp in exps:

    # Set up kernel herding params
    kernel_theta_type = exp[0]
    opt_manifold_type = exp[1]

    # Set up KR-ABC params
    kernel_y_type = "Sphere"

    errs =[]
    for idx,hyperparam in enumerate(random_hyperparams):
        print("------\nTrying hyperparam {} of {}\n-----".format(idx+1, len(random_hyperparams)))
        reg, bandwidth_theta = hyperparam
        # Run recursive kernel ABC
        theta_estimates, theta_herds = KernelRecursiveABC(y, n_iters, prior_sampler,
                                                          target_simulator, None,
                                                          kernel_y_type, kernel_theta_type,
                                                          opt_manifold_type, bandwidth_theta,
                                                          n_herd, reg, true_manifold,
                                                          true_theta=true_loc,
                                                          y_dist_func=y_dist_func,
                                                          project_func=y_project_func)
        # Compute error for theta_estimates
        err = sphere_dist(theta_estimates[-1, :].reshape(1, dims), true_loc)
        errs.append(err.reshape(-1)[0])

    print("-----------Done----------")
    best_param = random_hyperparams[np.argmin(errs)]
    print("Best reg: {}, best bandwidth: {}".format(best_param[0], best_param[1]))
    regs = random_hyperparams[:, 0]
    bands = random_hyperparams[:, 1]
    fig = plt.figure()
    scat = plt.scatter(regs, bands, c=errs, cmap="seismic")
    plt.xlim(min(regs)*0.1, max(regs)*10)
    plt.xscale('log')
    plt.colorbar(scat)
    plt.xlabel("Regularizer")
    plt.ylabel("Bandwidth")
    plt.title("Best Params: reg: {}, bandwidth: {}".format(best_param[0], best_param[1]))
    plt.savefig('tuning/{}Kernel_{}Optimization.png'.format(kernel_theta_type,
                                                     opt_manifold_type))

